# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Adapted from the silentcipher pip package (sony/silentcipher on HuggingFace).
import torch
import torch.nn as nn


class Layer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True
        )
        self.gate = nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True
        )
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return self.bn(self.conv(x) * torch.sigmoid(self.gate(x)))


class MsgDecoder(nn.Module):
    def __init__(self, message_dim=0, message_band_size=None, channel_dim=128, num_layers=10):
        super().__init__()
        assert message_band_size is not None
        self.message_band_size = message_band_size

        main = [
            nn.Dropout(0),
            Layer(dim_in=1, dim_out=channel_dim, kernel_size=3, stride=1, padding=1),
        ]
        for _ in range(num_layers - 2):
            main += [
                nn.Dropout(0),
                Layer(dim_in=channel_dim, dim_out=channel_dim, kernel_size=3, stride=1, padding=1),
            ]
        main += [
            nn.Dropout(0),
            Layer(dim_in=channel_dim, dim_out=message_dim, kernel_size=3, stride=1, padding=1),
        ]
        self.main = nn.Sequential(*main)
        self.linear = nn.Linear(self.message_band_size, 1)

    def forward(self, x):
        h = self.main(x[:, :, : self.message_band_size])
        h = self.linear(h.transpose(2, 3)).squeeze(3).unsqueeze(1)
        return h
