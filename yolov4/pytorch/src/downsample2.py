# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from .resblock import ResBlock


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(128)
        self.mish = Mish()

        self.c2 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.b2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm2d(64)

        self.res = ResBlock(ch=64, nblocks=2)

        self.c4 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(64)

        self.c5 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)
        self.b5 = nn.BatchNorm2d(128)

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.mish(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.mish(x2_b)

        x3 = self.c3(x1_m)
        x3_b = self.b3(x3)
        x3_m = self.mish(x3_b)

        r1 = self.res(x3_m)

        x4 = self.c4(r1)
        x4_b = self.b4(x4)
        x4_m = self.mish(x4_b)

        x4_m = torch.cat([x4_m, x2_m], dim=1)

        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.mish(x5_b)
        return x5_m
