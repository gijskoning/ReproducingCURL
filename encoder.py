import torch.nn
from torch import nn


class PixelEncoder(nn.Module):

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = torch.nn.Sequential()
        self.layers.add_module("conv2d_0", nn.Conv2d(obs_shape[0], num_filters, 3, stride=2))
        self.layers.add_module("relu_0", nn.ReLU())
        for i in range(1, num_layers):
            self.layers.add_module("conv2d_" + i, nn.Conv2d(num_filters, num_filters, 3))
            self.layers.add_module("relu_" + i, nn.ReLU())

        self.layers.add_module("flatten", nn.Flatten())
        self.layers.add_module("lin_0", nn.Linear(39200, 1024))
        self.layers.add_module("lin_1", nn.Linear(1024, feature_dim))
        self.layers.add_module("norm", nn.LayerNorm(feature_dim))
        self.layers.add_module("tanh", nn.Tanh())

    def forward(self, x):
        x /= 255.
        return self.layers.forward(x)


class IdentityEncoder(nn.Module):
    """
    Mock encoder
    """
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
        encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )
