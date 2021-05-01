from torch import nn


class PixelEncoder(nn.Module):

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        raise Exception("Needs to be implemented")

_AVAILABLE_ENCODERS = {'pixel': PixelEncoder}  # , 'identity': IdentityEncoder}


def make_encoder(
        encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )
