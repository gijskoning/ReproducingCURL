"""
This file was initially copied from https://github.com/denisyarats/pytorch_sac_ae
Changes were made to the following classes/functions:
- Actor:
    - remove encoder
- Critic remove encoder
- SacAeAgent -> SacCurlAgent:
    - remove decoder
    - add query_encoder
    - add key_encoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CurlEncoder(nn.Module):

    def __init__(self, encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, device):
        super().__init__()
        # init encoders
        self.query = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters).to(device)
        self.key = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters).to(device)
        self.key.load_state_dict(self.query.state_dict())

        # init bilinear similarity matrix
        self.W = nn.Parameter(torch.rand((encoder_feature_dim, encoder_feature_dim)).to(device))

    def similarity(self, x1, x2):
        """
        Computes the logits and stabilizes them.
        :param x1: querys in the latent space
        :param x2: keys in the latent space
        :return: logit matrix of size (B, B)
        """
        sim = torch.mm(x2, torch.mm(self.W, x1.T))
        return sim - torch.max(sim, dim=1)[0]


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(self, input_dim, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
            self, obs, compute_pi=True, compute_log_pi=True
    ):

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
            self, input_dim, action_shape, hidden_dim
    ):
        super().__init__()

        self.Q1 = QFunction(
            input_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            input_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class SacCurlAgent(object):
    """SAC+CURL algorithm."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            hidden_dim=1024,
            discount=0.99,
            init_temperature=0.1,
            alpha_lr=1e-4,
            alpha_beta=0.5,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.01,
            critic_target_update_freq=2,
            encoder_type='pixel',
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_beta=0.9,
            encoder_tau=0.05,
            num_layers=4,
            num_filters=32,
            batch_size=128
    ):
        self.obs_shape = obs_shape
        self.crop_size = self.obs_shape[-1]
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        # init the CURL encoder
        self.encoder = CurlEncoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, device
        )

        # init SAC nets
        self.actor = Actor(
            encoder_feature_dim, action_shape, hidden_dim, actor_log_std_min, actor_log_std_max
        ).to(device)

        self.critic = Critic(
            encoder_feature_dim, action_shape, hidden_dim
        ).to(device)

        self.critic_target = Critic(
            encoder_feature_dim, action_shape, hidden_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.query.parameters(), lr=encoder_lr, betas=(encoder_beta, 0.999)
        )
        self.contrastive_optimizer = torch.optim.Adam(
            [self.encoder.W], lr=encoder_lr, betas=(encoder_beta, 0.999)
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = utils.random_crop(np.expand_dims(obs, axis=0), self.crop_size)
            obs = torch.FloatTensor(obs).to(self.device)

            latent_vector = self.encoder.query(obs)
            mu, _, _, _ = self.actor(
                latent_vector, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = utils.random_crop(np.expand_dims(obs, axis=0), self.crop_size)
            obs = torch.FloatTensor(obs).to(self.device)

            latent_vector = self.encoder.query(obs)

            mu, pi, _, _ = self.actor(latent_vector, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            latent_vector = self.encoder.query(next_obs)

            _, policy_action, log_pi, _ = self.actor(latent_vector)
            target_Q1, target_Q2 = self.critic_target(latent_vector, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        latent_vector = self.encoder.query(obs)
        current_Q1, current_Q2 = self.critic(latent_vector, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        latent_vector = self.encoder.query(obs, detach=True)

        _, pi, log_pi, log_std = self.actor(latent_vector)
        actor_Q1, actor_Q2 = self.critic(latent_vector, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_encoder(self, obs, obs_other_augmentation):
        augmented_query = obs
        augmented_key = obs_other_augmentation
        latent_query = self.encoder.query(augmented_query)
        latent_key = self.encoder.key(augmented_key, detach=True)
        logits = self.encoder.similarity(latent_query, latent_key)

        # CURL paper uses .long() not sure why though
        labels = torch.arange(logits.shape[0]).long().to(self.device)

        loss = self.cross_entropy_loss(logits, labels)
        self.encoder_optimizer.zero_grad()
        self.contrastive_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.contrastive_optimizer.step()

        utils.soft_update_params(self.encoder.query, self.encoder.key, self.encoder_tau)

    def update(self, replay_buffer: utils.ReplayBuffer, L, step):
        obs, obs_other_augmentation, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_encoder(obs, obs_other_augmentation)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )

    def save(self, model_dir, step):
        torch.save(
            self.encoder.query.state_dict(), '%s/encoder_query_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.encoder.key.state_dict(), '%s/encoder_key_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.encoder.W.state_dict(), '%s/encoder_W_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.encoder.query.load_state_dict(
            torch.load('%s/encoder_query_%s.pt' % (model_dir, step))
        )
        self.encoder.key.load_state_dict(
            torch.load('%s/encoder_key_%s.pt' % (model_dir, step))
        )
        self.encoder.W.load_state_dict(
            torch.load('%s/encoder_W_%s.pt' % (model_dir, step))
        )
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
