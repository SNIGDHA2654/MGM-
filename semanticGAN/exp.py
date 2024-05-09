import sys
import os
sys.path.append('..')

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from models.stylegan2_seg import GeneratorSeg, Discriminator, MultiscaleDiscriminator, GANLoss
# from models.stylegan2_seg import GeneratorSeg
print(f"Imported the modules")
generator = GeneratorSeg(
        256, 512, 8, seg_dim=8,
        image_mode='RGB', channel_multiplier=2
    ).to(device)
print(f"Loaded the model to device : {device}")
