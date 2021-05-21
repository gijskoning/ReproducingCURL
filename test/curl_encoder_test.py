import torch
from sac_curl import CurlEncoder

# encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, device
encoder = CurlEncoder('pixel', (9, 84, 84), 50, 4, 32, 'cuda')
b = 5
x1 = torch.rand((b, 50)).to('cuda')
x2 = torch.rand((b, 50)).to('cuda')
print(encoder.similarity(x1, x2))
