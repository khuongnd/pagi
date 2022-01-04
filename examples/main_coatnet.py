import torch
from pagi.models.vits.coatnet import coatnet_0, CoAtNet

img = torch.randn(1, 3, 224, 224)
net = coatnet_0()
out = net(img)

num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D
block_types=['C', 'T', 'T', 'T']        # 'C' for MBConv, 'T' for Transformer

net = CoAtNet((224, 224), 3, num_blocks, channels, block_types=block_types)
out = net(img)