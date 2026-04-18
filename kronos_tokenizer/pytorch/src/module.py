# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/shiyu-coder/Kronos
# License: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class BinarySphericalQuantizer(nn.Module):
    def __init__(
        self,
        embed_dim,
        beta,
        gamma0,
        gamma,
        zeta,
        input_format="bchw",
        soft_entropy=True,
        group_size=9,
        persample_entropy_compute="analytical",
        cb_entropy_compute="group",
        l2_norm=True,
        inv_temperature=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.input_format = input_format
        self.num_groups = embed_dim // group_size
        self.group_size = group_size
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature

        self.register_buffer("basis", 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer("group_basis", 2 ** torch.arange(group_size - 1, -1, -1))

        group_codes = torch.arange(2**self.group_size)
        group_codebook = self._indexes_to_codes(group_codes, self.embed_dim).float()[
            :, -group_size:
        ]
        self.register_buffer("group_codebook", group_codebook, persistent=False)
        self.soft_entropy = soft_entropy

    def _indexes_to_codes(self, indices, embed_dim):
        basis = 2 ** torch.arange(embed_dim - 1, -1, -1)
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, basis), 2)
        return codes_non_centered * 2 - 1

    def quantize(self, z):
        zhat = torch.where(
            z > 0,
            torch.tensor(1, dtype=z.dtype, device=z.device),
            torch.tensor(-1, dtype=z.dtype, device=z.device),
        )
        return z + (zhat - z).detach()

    def forward(self, z, collect_metrics=True):
        zq = self.quantize(z)
        q_scale = 1.0 / (self.embed_dim**0.5) if self.l2_norm else 1.0
        zq = zq * q_scale

        if not collect_metrics:
            return zq, zq.new_zeros(()), {}

        indices = self.codes_to_indexes(zq.detach())
        used_codes = (
            torch.unique(indices, return_counts=False) if not self.training else None
        )

        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb_by_sample = (
                ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
            )
            persample_entropy = self._get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = torch.tensor(0.0, device=z.device)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        return (
            zq,
            commit_loss + self.zeta * entropy_penalty / self.inv_temperature,
            {
                "H": cb_entropy,
                "used_codes": used_codes,
                "indices": indices,
                "avg_prob": avg_prob,
            },
        )

    def soft_entropy_loss(self, z):
        group_code_book = self.group_codebook / (
            self.embed_dim**0.5 if self.l2_norm else 1
        )
        divided_z = rearrange(z, "... (g c) -> ... g c", c=self.group_size)
        distance = -2 * torch.einsum(
            "... g c, d c ->... g d", divided_z, group_code_book
        )
        prob = (-distance * self.inv_temperature).softmax(dim=-1)
        if self.persample_entropy_compute == "analytical":
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim**0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = (
                self._get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
            )
        else:
            per_sample_entropy = (
                self._get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
            )

        avg_prob = reduce(prob, "... g d ->g d", "mean")
        codebook_entropy = self._get_entropy(avg_prob, dim=-1, normalize=False)
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def _get_hard_per_sample_entropy(self, zb_by_sample):
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = -probs_per_dim * torch.log(probs_per_dim + 1e-8) - (
            1 - probs_per_dim
        ) * torch.log(1 - probs_per_dim + 1e-8)
        return persample_entropy.sum(-1).mean()

    def codes_to_indexes(self, zhat):
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self.basis), 2)
        return codes_non_centered * 2 - 1

    def _get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        return -(probs * torch.log(probs + 1e-8)).sum(dim=dim)


class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.codebook_dim = s1_bits + s2_bits
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.bsq = BinarySphericalQuantizer(
            self.codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size
        )

    def bits_to_indices(self, bits):
        bits = (bits >= 0).to(torch.long)
        indices = 2 ** torch.arange(
            0, bits.shape[-1], 1, dtype=torch.long, device=bits.device
        )
        return (bits * indices).sum(-1)

    def forward(self, z, half=False, collect_metrics=True):
        z = F.normalize(z, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
        if half:
            q_pre = quantized[:, :, : self.s1_bits]
            q_post = quantized[:, :, self.s1_bits :]
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len = q.shape[-2]
        t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.resid_dropout(self.out_proj(attn_output))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        ff_dim=1024,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(
            d_model, n_heads, attn_dropout_p, resid_dropout_p
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x
