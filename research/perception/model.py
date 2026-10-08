"""Small U-Net for floor-plan perception (research). Input: 1 x H x W ink in [0, 1] (1 = dark);
output: logits for CLASSES of synth_data.py. H, W multiples of 16."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 7


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base: int = 24, n_classes: int = N_CLASSES):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 12]
        self.enc = nn.ModuleList([_block(1, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3]),
                                  _block(c[3], c[4])])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < 4:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for i in reversed(range(4)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)
