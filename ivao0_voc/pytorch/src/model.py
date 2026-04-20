# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Model definition for ivao0/voc Attentionless Vocoder Streaming.

Ported from the usage snippet on the model card:
https://huggingface.co/ivao0/voc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2PreTrainedModel


class BufferConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous = None

    def forward(self, x):
        k = self.kernel_size[0]
        if self.previous is not None:
            x = torch.cat([self.previous, x], 2)
        else:
            if k == 3:
                x = F.pad(x, (2, 0), mode="replicate", value=0.0)
            elif k == 4:
                x = F.pad(x, (3, 0), mode="replicate", value=0.0)
            elif k == 7:
                x = F.pad(x, (6, 0), mode="replicate", value=0.0)
            elif k == 16:
                x = F.pad(x, (2, 0), mode="replicate", value=0.0)
        num_frames = int((x.shape[2] - self.kernel_size[0]) / self.stride[0]) + 1
        offset = num_frames * self.stride[0]
        self.previous = x[..., offset:]
        return super().forward(x)


class BufferConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial = None

    def forward(self, x):
        out = super().forward(x)
        out_len = out.shape[2]
        invalid_steps = self.kernel_size[0] - self.stride[0]
        if self.partial is not None:
            partial_len = self.partial.shape[-1]
            if self.bias is not None:
                out[..., :partial_len] += self.partial - self.bias[:, None]
            else:
                out[..., :partial_len] += self.partial
        self.partial = out[..., out_len - invalid_steps :]
        out = out[..., : out_len - invalid_steps]
        return out


class SEANetResnetBlock(nn.Module):
    def __init__(self, dim, kernel_sizes=(3, 1)):
        super().__init__()
        block = []
        for i, kernel_size in enumerate(kernel_sizes):
            block += [
                nn.ELU(),
                BufferConv1d(
                    dim if i == 0 else dim // 2,
                    dim // 2 if i == 0 else dim,
                    kernel_size=kernel_size,
                    bias=True,
                ),
            ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class SEANetEncoder(nn.Module):
    def __init__(
        self,
        channels=1,
        dimension=512,
        n_filters=64,
        ratios=(8, 6, 5, 4),
        kernel_size=7,
        last_kernel_size=3,
    ):
        super().__init__()
        self.ratios = list(reversed(ratios))
        mult = 1
        model = [BufferConv1d(channels, mult * n_filters, kernel_size, bias=True)]
        for ratio in self.ratios:
            model += [
                SEANetResnetBlock(mult * n_filters),
                nn.ELU(),
                BufferConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    bias=True,
                ),
            ]
            mult *= 2
        model += [
            nn.ELU(),
            BufferConv1d(mult * n_filters, dimension, last_kernel_size, bias=True),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    def __init__(
        self,
        channels=1,
        dimension=512,
        n_filters=64,
        ratios=(8, 6, 5, 4),
        kernel_size=7,
        last_kernel_size=3,
    ):
        super().__init__()
        mult = int(2 ** len(ratios))
        model = [BufferConv1d(dimension, mult * n_filters, kernel_size, bias=True)]
        for ratio in ratios:
            model += [
                nn.ELU(),
                BufferConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    bias=True,
                ),
                SEANetResnetBlock(mult * n_filters // 2),
            ]
            mult //= 2
        model += [
            nn.ELU(),
            BufferConv1d(n_filters, channels, last_kernel_size, bias=True),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CodeBook(nn.Module):
    def __init__(self, dim, codebook_size):
        super().__init__()
        self.register_buffer("_e", torch.zeros(codebook_size, dim))

    def encode(self, x):
        dist = torch.cdist(x.transpose(1, 2), self._e[None, :, :])
        return dist.argmin(2)

    def decode(self, codes):
        quantized = F.embedding(codes, self._e)
        return quantized.transpose(1, 2)


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_s = nn.Conv1d(512, 256, 1, bias=False)
        self.in_proj_a = nn.Conv1d(512, 256, 1, bias=False)
        self.out_proj_s = nn.Conv1d(256, 512, 1, bias=False)
        self.out_proj_a = nn.Conv1d(256, 512, 1, bias=False)
        self.layers = nn.ModuleList(
            [CodeBook(dim=256, codebook_size=2048) for _ in range(18)]
        )
        # RVQ codebook indices reused for higher fidelity (see model card).
        self._acoustic_books = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            17,
            17,
            17,
            17,
        ]

    def encode(self, x):
        indices = self.layers[0].encode(self.in_proj_s(x))
        all_indices = [indices[:, None, :]]
        x = self.in_proj_a(x)
        for cb in self._acoustic_books:
            indices = self.layers[cb].encode(x)
            x = x - self.layers[cb].decode(indices)
            all_indices.append(indices[:, None, :])
        return torch.cat(all_indices, 1)

    def decode(self, codes):
        semantic = self.layers[0].decode(codes[:, 0, :])
        acoustic = torch.zeros([1, 1], device=codes.device)
        for i, cb in enumerate(self._acoustic_books):
            acoustic = acoustic + self.layers[cb].decode(codes[:, i + 1, :])
        return self.out_proj_s(semantic) + self.out_proj_a(acoustic)


class VocAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fused_proj = nn.Parameter(torch.zeros(embed_dim, embed_dim))

    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(1, keepdims=True)
        return torch.matmul(x, self.fused_proj)


class VocTransformerLayer(nn.Module):
    def __init__(self, d_model=512, dim_feedforward=2048):
        super().__init__()
        self.self_attn = VocAttention(embed_dim=d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))


class VocTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(VocTransformerLayer() for _ in range(8))

    def forward(self, x):
        x = x.transpose(1, 2)
        for la in self.layers:
            x = la(x)
        return x.transpose(1, 2)


class Voc(Wav2Vec2PreTrainedModel):
    """Attentionless Vocoder Streaming based on kyutai/mimi."""

    def __init__(self, config):
        super().__init__(config=config)
        self.encoder_transformer = VocTransformer()
        self.decoder_transformer = VocTransformer()
        self.encoder = SEANetEncoder()
        self.decoder = SEANetDecoder()
        self.sample_rate = 24000
        self.quantizer = SplitResidualVectorQuantizer()
        self.downsample = BufferConv1d(
            512, 512, kernel_size=4, stride=2, groups=1, bias=False
        )
        self.upsample = BufferConvTranspose1d(
            512, 512, kernel_size=4, groups=512, stride=2, bias=False
        )
        self.frame_rate = 12.5
        self.encode_buffer = None

    def encode(self, x):
        """24 kHz audio (bs, 1, samples) -> codes (bs, 22, time)."""
        if self.encode_buffer is not None:
            x = torch.cat([self.encode_buffer, x], 2)
        bs, _, length = x.shape
        num_frames = int(length / 1920)
        leftover = x[:, :, (num_frames + 1) * 1920 :]
        self.encode_buffer = leftover if leftover.shape[2] > 0 else None
        if num_frames > 0:
            c = []
            for n in range(num_frames):
                e = self.encoder(x[:, :, n * 1920 : (n + 1) * 1920])
                e = self.encoder_transformer(e)
                e = self.downsample(e)
                c.append(self.quantizer.encode(e))
            return torch.cat(c, 2)
        return torch.empty(bs, 0, 22)

    def decode(self, c):
        """Codes (bs, 22, n_tokens) -> 24 kHz audio (bs, 1, n_tokens * 1920)."""
        hidden = []
        for i in range(c.shape[2]):
            x = self.quantizer.decode(c[:, :, i : i + 1])
            x = self.upsample(x)
            x = self.decoder_transformer(x)
            x = self.decoder(x)
            hidden.append(x)
        return torch.cat(hidden, 2)
