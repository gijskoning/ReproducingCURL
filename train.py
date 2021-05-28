"""
This file was initially copied from https://github.com/denisyarats/pytorch_sac_ae
Changes were made to the following classes/functions:
"""
import shutil
from datetime import datetime

import torch
import argparse
import os
import time
import json
import dmc2gym

from torch.utils.tensorboard import SummaryWriter

import utils
from logger import Logger
from video import VideoRecorder

from sac_curl import SacCurlAgent

args = None


def parse_args(_args=None):
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1e6, type=int)
    # train
    parser.add_argument('--agent', default='sac_curl', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')

    # Arguments added ourselves:
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--only_cpu', default=False, action='store_true')
    parser.add_argument('--load', default='', type=str)

    return parser.parse_args(_args)


def evaluate(env, agent, video, num_episodes, L, step):
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac_curl':
        return SacCurlAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            batch_size=args.batch_size
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main(_args=None):
    global args
    args = parse_args(_args)
    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)
    work_dir_old = "old_tmp"
    args.work_dir += "_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    if args.load != '':
        print(work_dir_old)
        utils.make_dir(work_dir_old)
        print(f"Continuing training {args.load}")
        shutil.copytree(args.load, work_dir_old +"/"+args.load[4:] +"_old_"+ datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        args.work_dir = args.load
    else:
        utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    buffer_dir = args.work_dir
    logger_dir = utils.make_dir(os.path.join(args.work_dir, 'logger'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.only_cpu else 'cpu')
    print("device used:", device)
    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        crop_size=args.image_size
    )
    if args.load != '':
        replay_buffer.load(args.load)
    shape = env.observation_space.shape
    agent = make_agent(
        # Change the image shape to accept cropped images. Keep the frame count
        obs_shape=(shape[0], args.image_size, args.image_size),
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )
    restarted = True
    if args.load != '':
        L: Logger = torch.load(args.load + "/logger/l.pt")
        L._sw = SummaryWriter(L.tb_dir)
        print(f"Continuing training from episode", L.episode, "and training step", L.step)
    else:
        restarted = False
        L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True

    start_time = time.time()

    if args.load != '':
        episode, episode_reward, done = L.episode, 0, True
    for step in range(L.step, args.num_train_steps):
        if done:
            if not restarted:
                if step > 0:
                    L.log('train/duration', time.time() - start_time, step)
                    start_time = time.time()
                    L.dump(step)

                # evaluate agent periodically
                if step % args.eval_freq == 0:
                    L.log('eval/episode', episode, step)
                    evaluate(env, agent, video, args.num_eval_episodes, L, step)
                    if args.save_model:
                        print("Saving model")
                        agent.save(model_dir, step)
                    if args.save_buffer:
                        print("Saving buffer and logger")
                        replay_buffer.save(buffer_dir)
                        # Cannot save Summary writer so removing it temporarily from Logger
                        sw = L._sw
                        L._sw = None
                        L.step = step
                        L.episode = episode

                        torch.save(L, logger_dir + "/l.pt")
                        torch.save(L, logger_dir + "/l.pt")
                        L._sw = sw

                L.log('train/episode_reward', episode_reward, step)
            restarted = False
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for i in range(num_updates):
                if i % 5 == 2:
                    print("init steps", i)
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    main()
