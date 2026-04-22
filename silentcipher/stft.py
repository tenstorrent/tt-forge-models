# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Adapted from the silentcipher pip package (sony/silentcipher on HuggingFace).
import torch


class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512):
        super().__init__()
        self.filter_length = filter_length
        self.hop_len = hop_length
        self.win_len = filter_length
        self.window = torch.hann_window(self.win_len)

    def transform(self, x):
        x = torch.nn.functional.pad(x, (0, self.win_len - x.shape[1] % self.win_len))
        fft = torch.stft(
            x,
            self.filter_length,
            self.hop_len,
            self.win_len,
            window=self.window.to(x.device),
            return_complex=True,
        )
        real_part, imag_part = fft.real, fft.imag
        squared = real_part**2 + imag_part**2
        additive_epsilon = torch.ones_like(squared) * (squared == 0).float() * 1e-24
        magnitude = torch.sqrt(squared + additive_epsilon) - torch.sqrt(
            additive_epsilon
        )
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data)
        ).float()
        return magnitude, phase
