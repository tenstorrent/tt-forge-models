# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Infinity model code.

"""

# =====================================================================
# 1. Common imports
# =====================================================================
import argparse
import math
import random
from collections import namedtuple
from contextlib import nullcontext
from functools import cache, partial
from math import ceil, log2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from timm.models.layers import DropPath
from torch import Tensor
from torch.amp import autocast
from torch.nn import Module
from torch.utils.checkpoint import checkpoint


# =====================================================================
# 2. Single-device distribution stubs (replaces infinity.utils.dist)
# =====================================================================
class _DistStub:
    """Minimal single-device stand-in for ``infinity.utils.dist``."""

    @staticmethod
    def get_world_size() -> int:
        return 1

    @staticmethod
    def get_rank() -> int:
        return 0

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def barrier() -> None:
        return None

    @staticmethod
    def allreduce(t, async_op=False):
        return None

    @staticmethod
    def is_master() -> bool:
        return True

    @staticmethod
    def is_visualizer() -> bool:
        return True


dist = _DistStub()


def for_visualize(func):
    """Single-device equivalent of ``infinity.utils.dist.for_visualize``."""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# =====================================================================
# 3. Flash-attn fallback + fused-op imports
# =====================================================================
from torch.nn.functional import scaled_dot_product_attention as slow_attn


def _flash_attn_varlen_kvpacked_func_fallback(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
):
    """Drop-in replacement for ``flash_attn_varlen_kvpacked_func`` using SDPA.

    Assumes uniform-length max_seqlen_q per batch (consistent with how
    Infinity's CrossAttention builds cu_seqlens_q).
    """
    H, C = q.shape[1], q.shape[2]
    B = cu_seqlens_q.numel() - 1
    Lq, Lk = int(max_seqlen_q), int(max_seqlen_k)
    q_b = q.view(B, Lq, H, C).transpose(1, 2).contiguous()
    kv_padded = kv.new_zeros(B, Lk, 2, H, C)
    mask_k = torch.zeros(B, Lk, dtype=torch.bool, device=q.device)
    cu_k = cu_seqlens_k.tolist()
    for i in range(B):
        s, e = cu_k[i], cu_k[i + 1]
        if e > s:
            kv_padded[i, : e - s] = kv[s:e]
            mask_k[i, : e - s] = True
    k_b = kv_padded[..., 0, :, :].transpose(1, 2).contiguous()
    v_b = kv_padded[..., 1, :, :].transpose(1, 2).contiguous()
    attn_mask = mask_k[:, None, None, :].expand(B, 1, Lq, Lk)
    out = slow_attn(
        q_b, k_b, v_b, attn_mask=attn_mask, scale=softmax_scale, dropout_p=dropout_p
    )
    return out.transpose(1, 2).reshape(B * Lq, H, C)


flash_attn_varlen_kvpacked_func = _flash_attn_varlen_kvpacked_func_fallback


try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm  # noqa: F401
    from flash_attn.ops.rms_norm import dropout_add_rms_norm  # noqa: F401
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func

    flash_fused_op_installed = True
except ImportError:
    dropout_add_layer_norm = dropout_add_rms_norm = fused_mlp_func = None
    flash_fused_op_installed = False

    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


# =====================================================================
# 4. Fused ada-norm helpers (infinity/models/fused_op.py)
# =====================================================================
@torch.compile(fullgraph=True)
def fused_rms_norm(x, weight, eps):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps))) * weight


@torch.compile(fullgraph=True)
def fused_ada_layer_norm(C, eps, x, scale, shift):
    x = x.float()
    x = F.layer_norm(input=x, normalized_shape=(C,), weight=None, bias=None, eps=eps)
    return x.mul(scale.add(1)).add_(shift)


@torch.compile(fullgraph=True)
def fused_ada_rms_norm(C, eps, x, scale, shift):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps))
    return x.mul(scale.add(1)).add_(shift)


# =====================================================================
# 5. Dynamic resolution tables
# =====================================================================
_vae_stride = 16
_ratio2hws = {
    1.000: [
        (1, 1),
        (2, 2),
        (4, 4),
        (6, 6),
        (8, 8),
        (12, 12),
        (16, 16),
        (20, 20),
        (24, 24),
        (32, 32),
        (40, 40),
        (48, 48),
        (64, 64),
    ],
    1.250: [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 4),
        (10, 8),
        (15, 12),
        (20, 16),
        (25, 20),
        (30, 24),
        (35, 28),
        (45, 36),
        (55, 44),
        (70, 56),
    ],
    1.333: [
        (1, 1),
        (2, 2),
        (4, 3),
        (8, 6),
        (12, 9),
        (16, 12),
        (20, 15),
        (24, 18),
        (28, 21),
        (36, 27),
        (48, 36),
        (60, 45),
        (72, 54),
    ],
    1.500: [
        (1, 1),
        (2, 2),
        (3, 2),
        (6, 4),
        (9, 6),
        (15, 10),
        (21, 14),
        (27, 18),
        (33, 22),
        (39, 26),
        (48, 32),
        (63, 42),
        (78, 52),
    ],
    1.750: [
        (1, 1),
        (2, 2),
        (3, 3),
        (7, 4),
        (11, 6),
        (14, 8),
        (21, 12),
        (28, 16),
        (35, 20),
        (42, 24),
        (56, 32),
        (70, 40),
        (84, 48),
    ],
    2.000: [
        (1, 1),
        (2, 2),
        (4, 2),
        (6, 3),
        (10, 5),
        (16, 8),
        (22, 11),
        (30, 15),
        (38, 19),
        (46, 23),
        (60, 30),
        (74, 37),
        (90, 45),
    ],
    2.500: [
        (1, 1),
        (2, 2),
        (5, 2),
        (10, 4),
        (15, 6),
        (20, 8),
        (25, 10),
        (30, 12),
        (40, 16),
        (50, 20),
        (65, 26),
        (80, 32),
        (100, 40),
    ],
    3.000: [
        (1, 1),
        (2, 2),
        (6, 2),
        (9, 3),
        (15, 5),
        (21, 7),
        (27, 9),
        (36, 12),
        (45, 15),
        (54, 18),
        (72, 24),
        (90, 30),
        (111, 37),
    ],
}
_predefined_t = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 21]

_full_ratio2hws = {}
for _r, _hws in _ratio2hws.items():
    _full_ratio2hws[_r] = _hws
    if _r != 1.000:
        _full_ratio2hws[int(1 / _r * 1000) / 1000] = [
            (item[1], item[0]) for item in _hws
        ]

dynamic_resolution_h_w = {}
for _r in _full_ratio2hws:
    dynamic_resolution_h_w[_r] = {}
    for _ind, _leng in enumerate([7, 10, 12, 13]):
        _h_div_w = _full_ratio2hws[_r][_leng - 1][0] / _full_ratio2hws[_r][_leng - 1][1]
        assert np.abs(_h_div_w - _r) < 0.01
        _pixel = (
            _full_ratio2hws[_r][_leng - 1][0] * _vae_stride,
            _full_ratio2hws[_r][_leng - 1][1] * _vae_stride,
        )
        if _ind == 0:
            _total_pixels = "0.06M"
        elif _ind == 1:
            _total_pixels = "0.25M"
        elif _ind == 2:
            _total_pixels = "0.60M"
        else:
            _total_pixels = "1M"
        _scales = _full_ratio2hws[_r][:_leng]
        _scales = [(t, h, w) for t, (h, w) in zip(_predefined_t, _scales)]
        dynamic_resolution_h_w[_r][_total_pixels] = {"pixel": _pixel, "scales": _scales}

h_div_w_templates = []
for _h_div_w in dynamic_resolution_h_w:
    h_div_w_templates.append(_h_div_w)
h_div_w_templates = np.array(h_div_w_templates)


# predefined_HW_Scales_dynamic from bsq_vae/dynamic_resolution.py (used by MultiScaleBSQ)
_full_ratio2hws_v2 = {}
for _r, _hws in _ratio2hws.items():
    _full_ratio2hws_v2[_r] = _hws
    _full_ratio2hws_v2[int(1 / _r * 1000) / 1000] = [
        (item[1], item[0]) for item in _hws
    ]

predefined_HW_Scales_dynamic = {}
for _r in _full_ratio2hws_v2:
    for _ind, _leng in enumerate([7, 10, 13]):
        _h, _w = (
            _full_ratio2hws_v2[_r][_leng - 1][0],
            _full_ratio2hws_v2[_r][_leng - 1][1],
        )
        predefined_HW_Scales_dynamic[(_h, _w)] = _full_ratio2hws_v2[_r][:_leng]


# =====================================================================
# 6. FlexAttn (only used when use_flex_attn=True)
# =====================================================================
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    _flex_attention_available = True
except ImportError:
    _flex_attention_available = False


def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _length_to_offsets(lengths, device):
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    return torch.cumsum(offsets, dim=-1)


def _generate_var_mask_mod(offsets):
    def _offsets_to_doc_ids_tensor(offsets):
        device = offsets.device
        counts = offsets[1:] - offsets[:-1]
        return torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.int32), counts
        )

    document_id = _offsets_to_doc_ids_tensor(offsets)

    def var_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        return same_doc | _causal_mask(b, h, q_idx, kv_idx)

    return var_mask_mod


def _generate_var_infer_mask_with_kv_cache(lengths):
    kv_len = sum(lengths)

    def var_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx < kv_len

    return var_mask_mod


class FlexAttn(nn.Module):
    def __init__(self, block_scales, mask_type, B, H, L, auto_padding=False):
        super().__init__()
        if not _flex_attention_available:
            raise NotImplementedError("flex attention needs pytorch 2.5.0+")
        self.auto_padding = auto_padding
        self.flex_attention = torch.compile(flex_attention)
        self.block_scales = block_scales
        self.lengths = [x * y * z for x, y, z in block_scales]
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        self.offsets = _length_to_offsets(self.lengths, device=_device)
        if self.offsets[-1] < L:
            self.offsets = torch.cat(
                (self.offsets, torch.tensor([L], device=_device)), dim=0
            )
        if mask_type == "var":
            self.mask_mod = _generate_var_mask_mod(self.offsets)
        elif mask_type == "causal":
            self.mask_mod = _causal_mask
        elif mask_type == "var_infer_mask_with_kv_cache":
            self.mask_mod = _generate_var_infer_mask_with_kv_cache(self.lengths)
        else:
            raise NotImplementedError(f"{mask_type} not supported")
        self.block_mask = create_block_mask(
            self.mask_mod, B=B, H=H, Q_LEN=L, KV_LEN=L, device=_device, _compile=True
        )

    def forward(self, q, k, v, scale=None):
        if self.auto_padding:
            q_pad_len = (128 - q.shape[-2] % 128) % 128
            kv_pad_len = (128 - k.shape[-2] % 128) % 128
            q_pad = F.pad(q, (0, 0, 0, q_pad_len))
            k_pad = F.pad(k, (0, 0, 0, kv_pad_len))
            v_pad = F.pad(v, (0, 0, 0, kv_pad_len))
            oup = self.flex_attention(
                q_pad.to(v_pad.dtype),
                k_pad.to(v.dtype),
                v_pad,
                block_mask=self.block_mask,
                scale=scale,
            )
            if q_pad_len > 0:
                oup = oup[:, :, :-q_pad_len]
        else:
            oup = self.flex_attention(
                q.to(v.dtype), k.to(v.dtype), v, block_mask=self.block_mask, scale=scale
            )
        return oup


# =====================================================================
# 7. BSQ-VAE
# =====================================================================

# ---- bsq_vae/conv.py ----
class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        cnn_type="2d",
        causal_offset=0,
        temporal_down=False,
    ):
        super().__init__()
        self.cnn_type = cnn_type
        self.slice_seq_len = 17
        if cnn_type == "2d":
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        if cnn_type == "3d":
            if not temporal_down:
                stride = (1, stride, stride)
            else:
                stride = (stride, stride, stride)
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride, padding=0
            )
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            self.padding = (kernel_size[0] - 1 + causal_offset, padding, padding)
        self.causal_offset = causal_offset
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.cnn_type == "2d":
            if x.ndim == 5:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                x = self.conv(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
                return x
            return self.conv(x)
        # 3D
        assert self.stride[0] in (1, 2)
        xs = []
        for i in range(0, x.shape[2], self.slice_seq_len + self.stride[0] - 1):
            st = i
            en = min(i + self.slice_seq_len, x.shape[2])
            _x = x[:, :, st:en, :, :]
            if i == 0:
                _x = F.pad(
                    _x,
                    (
                        self.padding[2],
                        self.padding[2],
                        self.padding[1],
                        self.padding[1],
                        self.padding[0],
                        0,
                    ),
                )
            else:
                padding_0 = self.kernel_size[0] - 1
                _x = F.pad(
                    _x,
                    (
                        self.padding[2],
                        self.padding[2],
                        self.padding[1],
                        self.padding[1],
                        padding_0,
                        0,
                    ),
                )
                _x[
                    :,
                    :,
                    :padding_0,
                    self.padding[1] : _x.shape[-2] - self.padding[1],
                    self.padding[2] : _x.shape[-1] - self.padding[2],
                ] += x[:, :, i - padding_0 : i, :, :]
            xs.append(self.conv(_x))
        return torch.cat(xs, dim=2)


# ---- bsq_vae/multiscale_bsq.py ----
Return = namedtuple(
    "Return", ["quantized", "indices", "bit_indices", "entropy_aux_loss"]
)
LossBreakdown = namedtuple(
    "LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"]
)


@cache
def _is_distributed():
    return False


def _exists(v):
    return v is not None


def _identity(t):
    return t


def _default(*args):
    for arg in args:
        if _exists(arg):
            return arg() if callable(arg) else arg
    return None


def _round_up_multiple(num, mult):
    return ceil(num / mult) * mult


def _pack_one(t, pattern):
    return pack([t], pattern)


def _unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def _l2norm(t):
    return F.normalize(t, dim=-1)


def _log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def _entropy(prob):
    return (-prob * _log(prob)).sum(dim=-1)


class CosineSimLinear(Module):
    def __init__(self, dim_in, dim_out, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=0)
        return (x @ w) * self.scale


def get_latent2scale_schedule(T, H, W, mode="original"):
    assert mode in ["original", "dynamic", "dense", "same1", "same2", "same3"]
    predefined_HW_Scales = {
        (32, 32): [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (6, 6),
            (9, 9),
            (13, 13),
            (18, 18),
            (24, 24),
            (32, 32),
        ],
        (16, 16): [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (8, 8),
            (10, 10),
            (13, 13),
            (16, 16),
        ],
        (64, 64): [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (7, 7),
            (9, 9),
            (12, 12),
            (16, 16),
            (21, 21),
            (27, 27),
            (36, 36),
            (48, 48),
            (64, 64),
        ],
        (36, 64): [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (6, 6),
            (9, 12),
            (13, 16),
            (18, 24),
            (24, 32),
            (32, 48),
            (36, 64),
        ],
    }
    if mode == "dynamic":
        predefined_HW_Scales.update(predefined_HW_Scales_dynamic)
    elif mode == "dense":
        predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 17)]
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [
            (20, 20),
            (24, 24),
            (28, 28),
            (32, 32),
        ]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [
            (40, 40),
            (48, 48),
            (56, 56),
            (64, 64),
        ]
    elif mode.startswith("same"):
        n = int(mode[len("same") :])
        predefined_HW_Scales[(16, 16)] = [(16, 16) for _ in range(n)]
        predefined_HW_Scales[(32, 32)] = [(32, 32) for _ in range(n)]
        predefined_HW_Scales[(64, 64)] = [(64, 64) for _ in range(n)]
    predefined_T_Scales = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 17, 17, 17, 17]
    patch_THW_shape_per_scale = predefined_HW_Scales[(H, W)]
    if len(predefined_T_Scales) < len(patch_THW_shape_per_scale):
        predefined_T_Scales += [predefined_T_Scales[-1]] * (
            len(patch_THW_shape_per_scale) - len(predefined_T_Scales)
        )
    return [
        (min(T, t), h, w)
        for (h, w), t in zip(
            patch_THW_shape_per_scale,
            predefined_T_Scales[: len(patch_THW_shape_per_scale)],
        )
    ]


class BSQLayerNorm(nn.Module):
    """LayerNorm variant used by MultiScaleBSQ (was ``LayerNorm`` in multiscale_bsq.py)."""

    def __init__(
        self,
        normalized_shape,
        norm_weight=False,
        eps=1e-6,
        data_format="channels_first",
    ):
        super().__init__()
        if norm_weight:
            self.weight = nn.Parameter(
                torch.ones(normalized_shape) / (normalized_shape**0.5)
            )
        else:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if x.ndim == 4:
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        elif x.ndim == 5:
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        else:
            raise ValueError("expected 4D or 5D input")
        return x


class MultiScaleBSQ(Module):
    """Multi-scale BSQ quantizer (see https://arxiv.org/abs/2406.07548)."""

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        soft_clamp_input_value=None,
        aux_loss=False,
        ln_before_quant=False,
        ln_init_by_sqrt=False,
        use_decay_factor=False,
        use_stochastic_depth=False,
        drop_rate=0.0,
        schedule_mode="original",
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        random_flip=False,
        flip_prob=0.5,
        flip_mode="stochastic",
        max_flip_lvl=1,
        random_flip_1lvl=False,
        flip_lvl_idx=None,
        drop_when_test=False,
        drop_lvl_idx=None,
        drop_lvl_num=0,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = int(log2(codebook_size))
        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.has_projections = requires_projection
        self.layernorm = (
            BSQLayerNorm(codebook_dim, norm_weight=ln_init_by_sqrt)
            if ln_before_quant
            else nn.Identity()
        )
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_rate = drop_rate
        self.remove_residual_detach = remove_residual_detach
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.flip_mode = flip_mode
        self.max_flip_lvl = max_flip_lvl
        self.random_flip_1lvl = random_flip_1lvl
        self.flip_lvl_idx = flip_lvl_idx
        assert not (random_flip and random_flip_1lvl)
        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0
        self.lfq = BSQ(
            dim=codebook_dim,
            codebook_scale=1 / np.sqrt(codebook_dim),
            soft_clamp_input_value=soft_clamp_input_value,
            **kwargs,
        )
        self.z_interplote_up = "trilinear"
        self.z_interplote_down = "area"
        self.use_decay_factor = use_decay_factor
        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant

    @property
    def codebooks(self):
        return self.lfq.codebook

    def get_codes_from_indices(self, indices_list):
        all_codes = [self.lfq.indices_to_codes(idx) for idx in indices_list]
        _, _, T, H, W = all_codes[-1].size()
        summed = 0
        for code in all_codes:
            summed += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        return self.project_out(reduce(codes, "q ... -> ...", "sum"))

    def flip_quant(self, x):
        assert self.flip_mode == "stochastic"
        flip_mask = torch.rand_like(x) < self.flip_prob
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x,
        scale_schedule=None,
        mask=None,
        return_all_codes=False,
        return_residual_norm_per_scale=False,
    ):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.size()
        if scale_schedule is None:
            if self.schedule_mode.startswith("same"):
                scale_num = int(self.schedule_mode[len("same") :])
                assert T == 1
                scale_schedule = [(1, H, W)] * scale_num
            else:
                scale_schedule = get_latent2scale_schedule(
                    T, H, W, mode=self.schedule_mode
                )
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.project_in(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.layernorm(x)
        quantized_out = 0.0
        residual = x
        (
            all_losses,
            all_indices,
            all_bit_indices,
            var_inputs,
            residual_norm_per_scale,
        ) = ([], [], [], [], [])
        out_fact = init_out_fact = 1.0
        if self.drop_when_test:
            drop_lvl_start, drop_lvl_end = (
                self.drop_lvl_idx,
                self.drop_lvl_idx + self.drop_lvl_num,
            )
        scale_num = len(scale_schedule)
        with autocast("cuda", enabled=False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):
                out_fact = (
                    max(0.1, out_fact) if self.use_decay_factor else init_out_fact
                )
                interpolate_residual = (
                    F.interpolate(
                        residual, size=(pt, ph, pw), mode=self.z_interplote_down
                    )
                    if (pt, ph, pw) != (T, H, W)
                    else residual
                )
                if return_residual_norm_per_scale:
                    residual_norm_per_scale.append(
                        (
                            torch.abs(interpolate_residual)
                            < 0.05 * self.lfq.codebook_scale
                        ).sum()
                        / interpolate_residual.numel()
                    )
                if (
                    self.training
                    and self.use_stochastic_depth
                    and random.random() < self.drop_rate
                ):
                    if (si == 0 and self.keep_first_quant) or (
                        si == scale_num - 1 and self.keep_last_quant
                    ):
                        quantized, indices, _, loss = self.lfq(interpolate_residual)
                        quantized = quantized * out_fact
                        all_indices.append(indices)
                        all_losses.append(loss)
                    else:
                        quantized = torch.zeros_like(interpolate_residual)
                elif self.drop_when_test and drop_lvl_start <= si < drop_lvl_end:
                    continue
                else:
                    quantized, indices, bit_indices, loss = self.lfq(
                        interpolate_residual
                    )
                    if self.random_flip and si < self.max_flip_lvl:
                        quantized = self.flip_quant(quantized)
                    if self.random_flip_1lvl and si == self.flip_lvl_idx:
                        quantized = self.flip_quant(quantized)
                    quantized = quantized * out_fact
                    all_indices.append(indices)
                if (pt, ph, pw) != (T, H, W):
                    quantized = F.interpolate(
                        quantized, size=(T, H, W), mode=self.z_interplote_up
                    ).contiguous()
                if self.remove_residual_detach:
                    residual = residual - quantized
                else:
                    residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized
                all_bit_indices.append(bit_indices)
                all_losses.append(loss)
                if si != scale_num - 1:
                    var_inputs.append(
                        F.interpolate(
                            quantized_out,
                            size=scale_schedule[si + 1],
                            mode=self.z_interplote_down,
                        ).contiguous()
                    )
                if self.use_decay_factor:
                    out_fact -= 0.1
        quantized_out = quantized_out.permute(0, 2, 3, 4, 1).contiguous()
        quantized_out = self.project_out(quantized_out)
        quantized_out = quantized_out.permute(0, 4, 1, 2, 3).contiguous()
        if quantized_out.size(2) == 1:
            quantized_out = quantized_out.squeeze(2)
        all_losses = torch.stack(all_losses, dim=-1)
        ret = (
            quantized_out,
            all_indices,
            all_bit_indices,
            residual_norm_per_scale,
            all_losses,
            var_inputs,
        )
        if not return_all_codes:
            return ret
        all_codes = self.get_codes_from_indices(all_indices)
        return (*ret, all_codes)


class BSQ(Module):
    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1.0,
        straight_through_activation=nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.0,
        frac_per_sample_entropy=1.0,
        has_projections=None,
        projection_has_bias=True,
        soft_clamp_input_value=None,
        cosine_sim_project_in=False,
        cosine_sim_project_in_scale=None,
        channel_first=None,
        experimental_softplus_entropy_loss=False,
        entropy_loss_offset=5.0,
        spherical=True,
        force_quantization_f32=True,
        inv_temperature=100.0,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        preserve_norm=False,
        new_quant=False,
        mask_out=False,
        use_out_phi=False,
        use_out_phi_res=False,
    ):
        super().__init__()
        assert _exists(dim) or _exists(codebook_size)
        assert not _exists(codebook_size) or log2(codebook_size).is_integer()
        codebook_size = _default(codebook_size, lambda: 2**dim)
        self.codebook_size = codebook_size
        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = _default(dim, codebook_dims)
        self.codebook_dims = codebook_dims
        has_projections = _default(has_projections, dim != codebook_dims)
        if cosine_sim_project_in:
            cosine_sim_project_in = _default(
                cosine_sim_project_in_scale, codebook_scale
            )
            project_in_klass = partial(CosineSimLinear, scale=cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias=projection_has_bias)
        self.project_in = (
            project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dims, dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections
        self.out_phi = (
            nn.Linear(codebook_dims, codebook_dims) if use_out_phi else nn.Identity()
        )
        self.use_out_phi_res = use_out_phi_res
        if self.use_out_phi_res:
            self.out_phi_scale = nn.Parameter(
                torch.zeros(codebook_dims), requires_grad=True
            )
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        keep_num_codebooks_dim = _default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.channel_first = channel_first
        self.activation = straight_through_activation
        if not spherical:
            raise ValueError("For BSQ, spherical must be True.")
        self.persample_entropy_compute = "analytical"
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.preserve_norm = preserve_norm
        self.new_quant = new_quant
        self.mask_out = mask_out
        assert 0 < frac_per_sample_entropy <= 1.0
        self.frac_per_sample_entropy = frac_per_sample_entropy
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.codebook_scale = codebook_scale
        self.commitment_loss_weight = commitment_loss_weight
        self.soft_clamp_input_value = soft_clamp_input_value
        assert (
            not _exists(soft_clamp_input_value)
            or soft_clamp_input_value >= codebook_scale
        )
        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss
        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        self.force_quantization_f32 = force_quantization_f32

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    def indices_to_codes(self, indices, label_type="int_label", project_out=True):
        assert label_type in ["int_label", "bit_label"]
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = _default(self.channel_first, is_img_or_video)
        if not self.keep_num_codebooks_dim:
            if label_type == "int_label":
                indices = rearrange(indices, "... -> ... 1")
            else:
                indices = indices.unsqueeze(-2)
        if label_type == "int_label":
            assert indices[..., None].int().min() > 0
            bits = ((indices[..., None].int() & self.mask) != 0).float()
        else:
            bits = indices
        codes = self.bits_to_codes(bits)
        codes = _l2norm(codes)
        codes = rearrange(codes, "... c d -> ... (c d)")
        if project_out:
            codes = self.project_out(codes)
        if should_transpose:
            codes = rearrange(codes, "b ... d -> b d ...")
        return codes

    def quantize(self, z):
        assert z.shape[-1] == self.codebook_dims
        zhat = torch.where(
            z > 0,
            torch.tensor(1, dtype=z.dtype, device=z.device),
            torch.tensor(-1, dtype=z.dtype, device=z.device),
        )
        return z + (zhat - z).detach()

    def quantize_new(self, z):
        assert z.shape[-1] == self.codebook_dims
        zhat = torch.where(
            z > 0,
            torch.tensor(1, dtype=z.dtype, device=z.device),
            torch.tensor(-1, dtype=z.dtype, device=z.device),
        )
        q_scale = 1.0 / (self.codebook_dims**0.5)
        zhat = q_scale * zhat
        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        p = torch.sigmoid(-4 * z / (self.codebook_dims**0.5) * self.inv_temperature)
        prob = torch.stack([p, 1 - p], dim=-1)
        per_sample_entropy = (
            self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        )
        avg_prob = reduce(prob, "... g d ->g d", "mean")
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        probs = (
            (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
            if normalize
            else count
        )
        return -(probs * torch.log(probs + 1e-8)).sum(dim=dim)

    def forward(self, x, return_loss_breakdown=False, mask=None, entropy_weight=0.1):
        is_img_or_video = x.ndim >= 4
        should_transpose = _default(self.channel_first, is_img_or_video)
        if should_transpose:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = _pack_one(x, "b * d")
        assert x.shape[-1] == self.dim
        x = self.project_in(x)
        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)
        x = _l2norm(x)
        force_f32 = self.force_quantization_f32
        quantization_context = (
            partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        )
        indices = None
        with quantization_context():
            if force_f32:
                orig_dtype = x.dtype
                x = x.float()
            if self.new_quant:
                quantized = self.quantize_new(x)
            bit_indices = (quantized > 0).int()
            entropy_penalty = persample_entropy = cb_entropy = self.zero
            commit_loss = self.zero
            if force_f32:
                x = x.type(orig_dtype)
        x = quantized
        x = rearrange(x, "b n c d -> b n (c d)")
        x = self.project_out(x)
        if should_transpose:
            x = _unpack_one(x, ps, "b * d")
            x = rearrange(x, "b ... d -> b d ...")
            bit_indices = _unpack_one(bit_indices, ps, "b * c d")
        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, "... 1 d -> ... d")
        aux_loss = (
            commit_loss * self.commitment_loss_weight
            + (self.zeta * entropy_penalty / self.inv_temperature) * entropy_weight
        )
        ret = Return(x, indices, bit_indices, aux_loss)
        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)


# ---- bsq_vae/flux_vqgan.py ----
_ptdtype = {None: torch.float32, "fp32": torch.float32, "bf16": torch.bfloat16}


class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type, norm_axis="spatial"):
        super().__init__()
        self.norm_axis = norm_axis
        assert norm_type in ["group", "batch", "no"]
        if norm_type == "group":
            if in_channels % 32 == 0:
                self.norm = nn.GroupNorm(
                    num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
                )
            elif in_channels % 24 == 0:
                self.norm = nn.GroupNorm(
                    num_groups=24, num_channels=in_channels, eps=1e-6, affine=True
                )
            else:
                raise NotImplementedError
        elif norm_type == "batch":
            self.norm = nn.SyncBatchNorm(in_channels, track_running_stats=False)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        if self.norm_axis == "spatial":
            if x.ndim == 4:
                return self.norm(x)
            B, C, T, H, W = x.shape
            x = rearrange(x, "B C T H W -> (B T) C H W")
            x = self.norm(x)
            return rearrange(x, "(B T) C H W -> B C T H W", T=T)
        return self.norm(x)


def _swish(x):
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group", cnn_param=None):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(
            in_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"]
        )
        self.q = Conv(in_channels, in_channels, kernel_size=1)
        self.k = Conv(in_channels, in_channels, kernel_size=1)
        self.v = Conv(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv(in_channels, in_channels, kernel_size=1)

    def attention(self, h_):
        B, _, T, _, _ = h_.shape
        h_ = self.norm(h_)
        h_ = rearrange(h_, "B C T H W -> (B T) C H W")
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)
        return rearrange(h_, "(b t) 1 (h w) c -> b c t h w", h=h, w=w, c=c, b=B, t=T)

    def forward(self, x):
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="group", cnn_param=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(
            in_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"]
        )
        ct1 = (
            "2d"
            if cnn_param["res_conv_2d"] in ("half", "full")
            else cnn_param["cnn_type"]
        )
        ct2 = "2d" if cnn_param["res_conv_2d"] == "full" else cnn_param["cnn_type"]
        self.conv1 = Conv(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type=ct1
        )
        self.norm2 = Normalize(
            out_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"]
        )
        self.conv2 = Conv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type=ct2
        )
        if in_channels != out_channels:
            self.nin_shortcut = Conv(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = self.conv1(_swish(self.norm1(x)))
        h = self.conv2(_swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(
        self, in_channels, cnn_type="2d", spatial_down=False, temporal_down=False
    ):
        super().__init__()
        assert spatial_down
        self.pad = (0, 1, 0, 1) if cnn_type == "2d" else (0, 1, 0, 1, 0, 0)
        self.conv = Conv(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            cnn_type=cnn_type,
            temporal_down=temporal_down,
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad, mode="constant", value=0))


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        cnn_type="2d",
        spatial_up=False,
        temporal_up=False,
        use_pxsl=False,
    ):
        super().__init__()
        if cnn_type == "2d":
            self.scale_factor, self.causal_offset = 2, 0
        else:
            assert spatial_up
            if temporal_up:
                self.scale_factor, self.causal_offset = (2, 2, 2), -1
            else:
                self.scale_factor, self.causal_offset = (1, 2, 2), 0
        self.use_pxsl = use_pxsl
        if self.use_pxsl:
            self.conv = Conv(
                in_channels,
                in_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                cnn_type=cnn_type,
                causal_offset=self.causal_offset,
            )
            self.pxsl = nn.PixelShuffle(2)
        else:
            self.conv = Conv(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                cnn_type=cnn_type,
                causal_offset=self.causal_offset,
            )

    def forward(self, x):
        if self.use_pxsl:
            return self.pxsl(self.conv(x))
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult,
        num_res_blocks,
        z_channels,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=4,
        norm_type="group",
        cnn_param=None,
        use_checkpoint=False,
        use_vae=True,
    ):
        super().__init__()
        self.max_down = np.log2(patch_size)
        self.temporal_max_down = np.log2(temporal_patch_size)
        self.temporal_down_offset = self.max_down - self.temporal_max_down
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.cnn_param = cnn_param
        self.use_checkpoint = use_checkpoint
        ct = "2d" if cnn_param["conv_in_out_2d"] == "yes" else cnn_param["cnn_type"]
        self.conv_in = Conv(
            in_channels, ch, kernel_size=3, stride=1, padding=1, cnn_type=ct
        )
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        norm_type=norm_type,
                        cnn_param=cnn_param,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            spatial_down = i_level < self.max_down
            temporal_down = (
                i_level < self.max_down and i_level >= self.temporal_down_offset
            )
            if spatial_down or temporal_down:
                down.downsample = Downsample(
                    block_in,
                    cnn_type=cnn_param["cnn_type"],
                    spatial_down=spatial_down,
                    temporal_down=temporal_down,
                )
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            norm_type=norm_type,
            cnn_param=cnn_param,
        )
        if cnn_param["cnn_attention"] == "yes":
            self.mid.attn_1 = AttnBlock(block_in, norm_type, cnn_param=cnn_param)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            norm_type=norm_type,
            cnn_param=cnn_param,
        )
        self.norm_out = Normalize(
            block_in, norm_type, norm_axis=cnn_param["cnn_norm_axis"]
        )
        out_ct = "2d" if cnn_param["conv_inner_2d"] == "yes" else cnn_param["cnn_type"]
        self.conv_out = Conv(
            block_in,
            (int(use_vae) + 1) * z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            cnn_type=out_ct,
        )

    def forward(self, x, return_hidden=False):
        if not self.use_checkpoint:
            return self._forward(x, return_hidden=return_hidden)
        return checkpoint(self._forward, x, return_hidden, use_reentrant=False)

    def _forward(self, x, return_hidden=False):
        h0 = self.conv_in(x)
        hs = [h0]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        hs_mid = [h]
        h = self.mid.block_1(h)
        if self.cnn_param["cnn_attention"] == "yes":
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        hs_mid.append(h)
        h = self.conv_out(_swish(self.norm_out(h)))
        return (h, hs, hs_mid) if return_hidden else h


class Decoder(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult,
        num_res_blocks,
        z_channels,
        out_ch=3,
        patch_size=8,
        temporal_patch_size=4,
        norm_type="group",
        cnn_param=None,
        use_checkpoint=False,
        use_freq_dec=False,
        use_pxsf=False,
    ):
        super().__init__()
        self.max_up = np.log2(patch_size)
        self.temporal_max_up = np.log2(temporal_patch_size)
        self.temporal_up_offset = self.max_up - self.temporal_max_up
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.ffactor = 2 ** (self.num_resolutions - 1)
        self.cnn_param = cnn_param
        self.use_checkpoint = use_checkpoint
        self.use_freq_dec = use_freq_dec
        self.use_pxsf = use_pxsf
        block_in = ch * ch_mult[self.num_resolutions - 1]
        in_ct = "2d" if cnn_param["conv_inner_2d"] == "yes" else cnn_param["cnn_type"]
        self.conv_in = Conv(
            z_channels, block_in, kernel_size=3, stride=1, padding=1, cnn_type=in_ct
        )
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            norm_type=norm_type,
            cnn_param=cnn_param,
        )
        if cnn_param["cnn_attention"] == "yes":
            self.mid.attn_1 = AttnBlock(
                block_in, norm_type=norm_type, cnn_param=cnn_param
            )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            norm_type=norm_type,
            cnn_param=cnn_param,
        )
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        norm_type=norm_type,
                        cnn_param=cnn_param,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            spatial_up = 1 <= i_level <= self.max_up
            temporal_up = (
                1 <= i_level <= self.max_up and i_level >= self.temporal_up_offset + 1
            )
            if spatial_up or temporal_up:
                up.upsample = Upsample(
                    block_in,
                    cnn_type=cnn_param["cnn_type"],
                    spatial_up=spatial_up,
                    temporal_up=temporal_up,
                    use_pxsl=self.use_pxsf,
                )
            self.up.insert(0, up)
        self.norm_out = Normalize(
            block_in, norm_type, norm_axis=cnn_param["cnn_norm_axis"]
        )
        out_ct = "2d" if cnn_param["conv_in_out_2d"] == "yes" else cnn_param["cnn_type"]
        self.conv_out = Conv(
            block_in, out_ch, kernel_size=3, stride=1, padding=1, cnn_type=out_ct
        )

    def forward(self, z):
        if not self.use_checkpoint:
            return self._forward(z)
        return checkpoint(self._forward, z, use_reentrant=False)

    def _forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        if self.cnn_param["cnn_attention"] == "yes":
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
        return self.conv_out(_swish(self.norm_out(h)))


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cnn_param = dict(
            cnn_type=args.cnn_type,
            conv_in_out_2d=args.conv_in_out_2d,
            res_conv_2d=args.res_conv_2d,
            cnn_attention=args.cnn_attention,
            cnn_norm_axis=args.cnn_norm_axis,
            conv_inner_2d=args.conv_inner_2d,
        )
        self.encoder = Encoder(
            ch=args.base_ch,
            ch_mult=args.encoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            cnn_param=cnn_param,
            use_checkpoint=args.use_checkpoint,
            use_vae=args.use_vae,
        )
        self.decoder = Decoder(
            ch=args.base_ch,
            ch_mult=args.decoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            cnn_param=cnn_param,
            use_checkpoint=args.use_checkpoint,
            use_freq_dec=args.use_freq_dec,
            use_pxsf=args.use_pxsf,
        )
        self.z_drop = nn.Dropout(args.z_drop)
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.codebook_dim = self.embed_dim = args.codebook_dim
        self.gan_feat_weight = args.gan_feat_weight
        self.video_perceptual_weight = args.video_perceptual_weight
        self.recon_loss_type = args.recon_loss_type
        self.l1_weight = args.l1_weight
        self.use_vae = args.use_vae
        self.kl_weight = args.kl_weight
        self.lfq_weight = args.lfq_weight
        self.image_gan_weight = args.image_gan_weight
        self.video_gan_weight = args.video_gan_weight
        self.perceptual_weight = args.perceptual_weight
        self.flux_weight = args.flux_weight
        self.cycle_weight = args.cycle_weight
        self.cycle_feat_weight = args.cycle_feat_weight
        self.cycle_gan_weight = args.cycle_gan_weight
        self.flux_image_encoder = None
        if not args.use_vae:
            assert args.quantizer_type == "MultiScaleBSQ"
            self.quantizer = MultiScaleBSQ(
                dim=args.codebook_dim,
                codebook_size=args.codebook_size,
                entropy_loss_weight=args.entropy_loss_weight,
                diversity_gamma=args.diversity_gamma,
                preserve_norm=args.preserve_norm,
                ln_before_quant=args.ln_before_quant,
                ln_init_by_sqrt=args.ln_init_by_sqrt,
                commitment_loss_weight=args.commitment_loss_weight,
                new_quant=args.new_quant,
                use_decay_factor=args.use_decay_factor,
                mask_out=args.mask_out,
                use_stochastic_depth=args.use_stochastic_depth,
                drop_rate=args.drop_rate,
                schedule_mode=args.schedule_mode,
                keep_first_quant=args.keep_first_quant,
                keep_last_quant=args.keep_last_quant,
                remove_residual_detach=args.remove_residual_detach,
                use_out_phi=args.use_out_phi,
                use_out_phi_res=args.use_out_phi_res,
                random_flip=args.random_flip,
                flip_prob=args.flip_prob,
                flip_mode=args.flip_mode,
                max_flip_lvl=args.max_flip_lvl,
                random_flip_1lvl=args.random_flip_1lvl,
                flip_lvl_idx=args.flip_lvl_idx,
                drop_when_test=args.drop_when_test,
                drop_lvl_idx=args.drop_lvl_idx,
                drop_lvl_num=args.drop_lvl_num,
            )
            self.quantize = self.quantizer
            self.vocab_size = args.codebook_size

    def forward(self, x):
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1
        enc_dtype = _ptdtype[self.args.encoder_dtype]
        with torch.amp.autocast(
            "cuda", dtype=enc_dtype, enabled=torch.cuda.is_available()
        ):
            h, hs, hs_mid = self.encoder(x, return_hidden=True)
        hs = [_h.detach() for _h in hs]
        hs_mid = [_h.detach() for _h in hs_mid]
        h = h.to(dtype=torch.float32)
        z, all_indices, _, _, all_loss, _ = self.quantizer(h)
        x_recon = self.decoder(z)
        return x_recon, {
            "commitment_loss": torch.mean(all_loss) * self.lfq_weight,
            "encodings": all_indices,
        }

    def encode_for_raw_features(
        self, x, scale_schedule, return_residual_norm_per_scale=False
    ):
        enc_dtype = _ptdtype[self.args.encoder_dtype]
        with torch.amp.autocast(
            "cuda", dtype=enc_dtype, enabled=torch.cuda.is_available()
        ):
            h, hs, hs_mid = self.encoder(x, return_hidden=True)
        hs = [_h.detach() for _h in hs]
        hs_mid = [_h.detach() for _h in hs_mid]
        return h.to(dtype=torch.float32), hs, hs_mid

    def encode(self, x, scale_schedule, return_residual_norm_per_scale=False):
        h, hs, hs_mid = self.encode_for_raw_features(
            x, scale_schedule, return_residual_norm_per_scale
        )
        (
            z,
            all_indices,
            all_bit_indices,
            residual_norm_per_scale,
            all_loss,
            var_input,
        ) = self.quantizer(
            h,
            scale_schedule=scale_schedule,
            return_residual_norm_per_scale=return_residual_norm_per_scale,
        )
        return h, z, all_indices, all_bit_indices, residual_norm_per_scale, var_input

    def decode(self, z):
        return torch.clamp(self.decoder(z), min=-1, max=1)

    def decode_from_indices(self, all_indices, scale_schedule, label_type):
        summed = 0
        for idx_Bl in all_indices:
            codes = self.quantizer.lfq.indices_to_codes(idx_Bl, label_type)
            summed += F.interpolate(
                codes, size=scale_schedule[-1], mode=self.quantizer.z_interplote_up
            )
        assert summed.shape[-3] == 1
        return summed, torch.clamp(self.decoder(summed.squeeze(-3)), min=-1, max=1)


# ---- bsq_vae/vae.py ----
def _load_cnn(model, state_dict, prefix, expand=False, use_linear=False):
    delete_keys, loaded_keys = [], []
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        _key = key[len(prefix) :]
        if _key in model.state_dict():
            if use_linear and any(
                t in key
                for t in (".q.weight", ".k.weight", ".v.weight", ".proj_out.weight")
            ):
                load_weights = state_dict[key].squeeze()
            elif _key.endswith(".conv.weight") and expand:
                if model.state_dict()[_key].shape == state_dict[key].shape:
                    load_weights = state_dict[key]
                else:
                    _expand_dim = model.state_dict()[_key].shape[2]
                    load_weights = (
                        state_dict[key].unsqueeze(2).repeat(1, 1, _expand_dim, 1, 1)
                    )
            else:
                load_weights = state_dict[key]
            model.state_dict()[_key].copy_(load_weights)
            delete_keys.append(key)
            loaded_keys.append(prefix + _key)
        conv_list = (
            ["conv"]
            if use_linear
            else ["conv", ".q.", ".k.", ".v.", ".proj_out.", ".nin_shortcut."]
        )
        if any(k in _key for k in conv_list):
            for suffix, mid in ((".weight", ".conv.weight"), (".bias", ".conv.bias")):
                if _key.endswith(suffix):
                    conv_key = _key.replace(suffix, mid)
                    if conv_key in model.state_dict():
                        if (
                            suffix == ".weight"
                            and model.state_dict()[conv_key].shape
                            != state_dict[key].shape
                        ):
                            _expand_dim = model.state_dict()[conv_key].shape[2]
                            lw = (
                                state_dict[key]
                                .unsqueeze(2)
                                .repeat(1, 1, _expand_dim, 1, 1)
                            )
                        else:
                            lw = state_dict[key]
                        model.state_dict()[conv_key].copy_(lw)
                        delete_keys.append(key)
                        loaded_keys.append(prefix + conv_key)
        if "norm" in _key:
            for suffix, mid in ((".weight", ".norm.weight"), (".bias", ".norm.bias")):
                if _key.endswith(suffix):
                    norm_key = _key.replace(suffix, mid)
                    if norm_key in model.state_dict():
                        model.state_dict()[norm_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix + norm_key)
    for key in delete_keys:
        del state_dict[key]
    return model, state_dict, loaded_keys


def vae_model(
    vqgan_ckpt,
    schedule_mode,
    codebook_dim,
    codebook_size,
    test_mode=True,
    patch_size=16,
    encoder_ch_mult=[1, 2, 4, 4, 4],
    decoder_ch_mult=[1, 2, 4, 4, 4],
):
    args = argparse.Namespace(
        vqgan_ckpt=vqgan_ckpt,
        sd_ckpt=None,
        inference_type="image",
        save="./imagenet_val_bsq",
        save_prediction=True,
        image_recon4video=False,
        junke_old=False,
        device="cuda",
        max_steps=1e6,
        log_every=1,
        visu_every=1000,
        ckpt_every=1000,
        default_root_dir="",
        compile="no",
        ema="no",
        lr=1e-4,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=0,
        optim_type="Adam",
        disc_optim_type=None,
        lr_min=0.0,
        warmup_lr_init=0.0,
        max_grad_norm=1.0,
        max_grad_norm_disc=1.0,
        disable_sch=False,
        patch_size=patch_size,
        temporal_patch_size=4,
        embedding_dim=256,
        codebook_dim=codebook_dim,
        num_quantizers=8,
        quantizer_type="MultiScaleBSQ",
        use_vae=False,
        use_freq_enc=False,
        use_freq_dec=False,
        preserve_norm=False,
        ln_before_quant=False,
        ln_init_by_sqrt=False,
        use_pxsf=False,
        new_quant=True,
        use_decay_factor=False,
        mask_out=False,
        use_stochastic_depth=False,
        drop_rate=0.0,
        schedule_mode=schedule_mode,
        lr_drop=None,
        lr_drop_rate=0.1,
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        use_out_phi=False,
        use_out_phi_res=False,
        use_lecam_reg=False,
        lecam_weight=0.05,
        perceptual_model="vgg16",
        base_ch_disc=64,
        random_flip=False,
        flip_prob=0.5,
        flip_mode="stochastic",
        max_flip_lvl=1,
        not_load_optimizer=False,
        use_lecam_reg_zero=False,
        freeze_encoder=False,
        rm_downsample=False,
        random_flip_1lvl=False,
        flip_lvl_idx=0,
        drop_when_test=False,
        drop_lvl_idx=0,
        drop_lvl_num=1,
        disc_version="v1",
        magvit_disc=False,
        sigmoid_in_disc=False,
        activation_in_disc="leaky_relu",
        apply_blur=False,
        apply_noise=False,
        dis_warmup_steps=0,
        dis_lr_multiplier=1.0,
        dis_minlr_multiplier=False,
        disc_channels=64,
        disc_layers=3,
        discriminator_iter_start=0,
        disc_pretrain_iter=0,
        disc_optim_steps=1,
        disc_warmup=0,
        disc_pool="no",
        disc_pool_size=1000,
        advanced_disc=False,
        recon_loss_type="l1",
        video_perceptual_weight=0.0,
        image_gan_weight=1.0,
        video_gan_weight=1.0,
        image_disc_weight=0.0,
        video_disc_weight=0.0,
        l1_weight=4.0,
        gan_feat_weight=0.0,
        perceptual_weight=0.0,
        kl_weight=0.0,
        lfq_weight=0.0,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1,
        norm_type="group",
        disc_loss_type="hinge",
        use_checkpoint=False,
        precision="fp32",
        encoder_dtype="fp32",
        upcast_attention="",
        upcast_tf32=False,
        tokenizer="flux",
        pretrained=None,
        pretrained_mode="full",
        inflation_pe=False,
        init_vgen="no",
        no_init_idis=False,
        init_idis="keep",
        init_vdis="no",
        enable_nan_detector=False,
        turn_on_profiler=False,
        profiler_scheduler_wait_steps=10,
        debug=True,
        video_logger=False,
        bytenas="",
        username="",
        seed=1234,
        vq_to_vae=False,
        load_not_strict=False,
        zero=0,
        bucket_cap_mb=40,
        manual_gc_interval=1000,
        data_path=[""],
        data_type=[""],
        dataset_list=["imagenet"],
        fps=-1,
        dataaug="resizecrop",
        multi_resolution=False,
        random_bucket_ratio=0.0,
        sequence_length=16,
        resolution=[256, 256],
        batch_size=[1],
        num_workers=0,
        image_channels=3,
        codebook_size=codebook_size,
        codebook_l2_norm=True,
        codebook_show_usage=True,
        commit_loss_beta=0.25,
        entropy_loss_ratio=0.0,
        base_ch=128,
        num_res_blocks=2,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        dropout_p=0.0,
        cnn_type="2d",
        cnn_version="v1",
        conv_in_out_2d="no",
        conv_inner_2d="no",
        res_conv_2d="no",
        cnn_attention="no",
        cnn_norm_axis="spatial",
        flux_weight=0,
        cycle_weight=0,
        cycle_feat_weight=0,
        cycle_gan_weight=0,
        cycle_loop=0,
        z_drop=0.0,
    )
    vae = AutoEncoder(args)
    if isinstance(vqgan_ckpt, str):
        state_dict = torch.load(
            args.vqgan_ckpt, map_location=torch.device("cpu"), weights_only=True
        )
    else:
        state_dict = args.vqgan_ckpt
    if state_dict:
        key = "ema" if args.ema == "yes" else "vae"
        vae, _, _ = _load_cnn(vae, state_dict[key], prefix="", expand=False)
    if test_mode:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
    return vae


# =====================================================================
# 8. Transformer blocks (infinity/models/basic.py)
# =====================================================================


def precompute_rope2d_freqs_grid(
    dim,
    dynamic_resolution_h_w,
    rope2d_normalized_by_hw,
    pad_to_multiplier=1,
    max_height=2048 // 16,
    max_width=2048 // 16,
    base=10000.0,
    device=None,
    scaling_factor=1.0,
):
    half_dim = dim // 2
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, half_dim, 2, dtype=torch.int64).float().to(device)
            / half_dim
        )
    )
    t_height = (
        torch.arange(max_height, device=device, dtype=torch.int64).type_as(inv_freq)
        / scaling_factor
    )
    t_width = (
        torch.arange(max_width, device=device, dtype=torch.int64).type_as(inv_freq)
        / scaling_factor
    )
    freqs_height = torch.outer(t_height, inv_freq)
    freqs_width = torch.outer(t_width, inv_freq)
    freqs_grid_map = torch.concat(
        [
            freqs_height[:, None, :].expand(-1, max_width, -1),
            freqs_width[None, :, :].expand(max_height, -1, -1),
        ],
        dim=-1,
    )
    freqs_grid_map = torch.stack(
        [torch.cos(freqs_grid_map), torch.sin(freqs_grid_map)], dim=0
    )
    rope2d_freqs_grid = {}
    for h_div_w in dynamic_resolution_h_w:
        scale_schedule = dynamic_resolution_h_w[h_div_w]["1M"]["scales"]
        _, ph, pw = scale_schedule[-1]
        max_edge = freqs_grid_map.shape[1]
        if ph >= pw:
            uph, upw = max_edge, int(max_edge / ph * pw)
        else:
            uph, upw = int(max_edge / pw * ph), max_edge
        rope_cache_list = []
        for (_, ph, pw) in scale_schedule:
            ph_mul_pw = ph * pw
            if rope2d_normalized_by_hw == 1:
                rope_cache = F.interpolate(
                    freqs_grid_map[:, :uph, :upw, :].permute([0, 3, 1, 2]),
                    size=(ph, pw),
                    mode="bilinear",
                    align_corners=True,
                )
                rope_cache = rope_cache.permute([0, 2, 3, 1])
            elif rope2d_normalized_by_hw == 2:
                _, uph, upw = scale_schedule[-1]
                indices = (
                    torch.stack(
                        [
                            (torch.arange(ph) * (uph / ph))
                            .reshape(ph, 1)
                            .expand(ph, pw),
                            (torch.arange(pw) * (upw / pw))
                            .reshape(1, pw)
                            .expand(ph, pw),
                        ],
                        dim=-1,
                    )
                    .round()
                    .int()
                    .reshape(-1, 2)
                )
                rope_cache = freqs_grid_map[:, indices[:, 0], indices[:, 1], :].reshape(
                    2, ph, pw, -1
                )
            elif rope2d_normalized_by_hw == 0:
                rope_cache = freqs_grid_map[:, :ph, :pw, :]
            else:
                raise ValueError
            rope_cache_list.append(rope_cache.reshape(2, ph_mul_pw, -1))
        cat_rope_cache = torch.cat(rope_cache_list, 1)
        if cat_rope_cache.shape[1] % pad_to_multiplier:
            pad = torch.zeros(
                2,
                pad_to_multiplier - cat_rope_cache.shape[1] % pad_to_multiplier,
                half_dim,
            )
            cat_rope_cache = torch.cat([cat_rope_cache, pad], dim=1)
        cat_rope_cache = cat_rope_cache[:, None, None, None]
        for pn in dynamic_resolution_h_w[h_div_w]:
            scale_schedule = dynamic_resolution_h_w[h_div_w][pn]["scales"]
            tmp = [(1, h, w) for _, h, w in scale_schedule]
            rope2d_freqs_grid[str(tuple(tmp))] = cat_rope_cache
    return rope2d_freqs_grid


def apply_rotary_emb(
    q,
    k,
    scale_schedule,
    rope2d_freqs_grid,
    pad_to_multiplier,
    rope2d_normalized_by_hw,
    scale_ind,
):
    qk = torch.stack((q, k), dim=0)
    device_type = qk.device.type
    device_type = (
        device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
        seq_len = qk.shape[3]
        start = 0
        if scale_ind >= 1:
            start = int(
                np.sum(
                    [item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]]
                )
            )
        rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[
            str(tuple(scale_schedule))
        ].to(qk.device)
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][
            :, :, :, :, start : start + seq_len
        ].to(dtype=qk.dtype)
        qk = qk.reshape(*qk.shape[:-1], -1, 2)
        qk = torch.stack(
            [
                rope_cache[0] * qk[..., 0] - rope_cache[1] * qk[..., 1],
                rope_cache[1] * qk[..., 0] + rope_cache[0] * qk[..., 1],
            ],
            dim=-1,
        )
        qk = qk.reshape(*qk.shape[:-2], -1)
        q, k = qk.unbind(dim=0)
    return q, k


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C, self.eps, self.elementwise_affine = C, eps, elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer("weight", torch.ones(C))

    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)


def _get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_mlp=False,
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = _get_dropout_layer(drop)
        self.heuristic = -1

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                    return_residual=False,
                    checkpoint_lvl=0,
                    heuristic=self.heuristic,
                    process_group=None,
                )
            )
        return self.drop(self.fc2(self.act(self.fc1(x))))


class FFNSwiGLU(nn.Module):
    def __init__(
        self, in_features, hidden_features, out_features=None, drop=0.0, fused_mlp=False
    ):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256
        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = _get_dropout_layer(drop)

    def forward(self, x):
        return self.drop(self.fc2(F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x))))


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        proj_drop=0.0,
        tau=1,
        cos_attn=False,
        customized_flash_attn=True,
        use_flex_attn=False,
        batch_size=2,
        pad_to_multiplier=1,
        rope2d_normalized_by_hw=0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (
                (1, 1, self.num_heads, 1)
                if self.using_flash
                else (1, self.num_heads, 1, 1)
            )
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=size, fill_value=4.0).log(), requires_grad=True
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = _get_dropout_layer(proj_drop)
        self.caching = False
        self.cached_k = None
        self.cached_v = None
        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

    def kv_caching(self, enable):
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def forward(
        self,
        x,
        attn_bias_or_two_vector,
        attn_fn=None,
        scale_schedule=None,
        rope2d_freqs_grid=None,
        scale_ind=0,
    ):
        B, L, C = x.shape
        qkv = F.linear(
            x,
            self.mat_qkv.weight,
            torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        if self.using_flash:
            q, k, v = qkv.unbind(dim=2)
            L_dim = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            L_dim = 2
        if self.cos_attn:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()
            v = v.contiguous()
        else:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        if rope2d_freqs_grid is not None:
            q, k = apply_rotary_emb(
                q,
                k,
                scale_schedule,
                rope2d_freqs_grid,
                self.pad_to_multiplier,
                self.rope2d_normalized_by_hw,
                scale_ind,
            )
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
        if self.using_flash:
            raise RuntimeError(
                "customized flash_attn_func is not vendored; instantiate with customized_flash_attn=False."
            )
        if self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            oup = (
                slow_attn(
                    query=q,
                    key=k,
                    value=v,
                    scale=self.scale,
                    attn_mask=attn_bias_or_two_vector,
                    dropout_p=0,
                )
                .transpose(1, 2)
                .reshape(B, L, C)
            )
        return self.proj_drop(self.proj(oup))


class CrossAttention(nn.Module):
    def __init__(
        self,
        for_attn_pool=False,
        embed_dim=768,
        kv_dim=4096,
        num_heads=12,
        proj_drop=0.0,
        cos_attn=False,
    ):
        cos_attn = False
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                requires_grad=True,
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim * 2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = _get_dropout_layer(proj_drop)

    def forward(self, q, ca_kv):
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        kv_compact = F.linear(
            kv_compact, self.mat_kv.weight, torch.cat((self.zero_k_bias, self.v_bias))
        ).view(N, 2, self.num_heads, self.head_dim)
        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        if self.cos_attn:
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)
        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        cu_seqlens_q = torch.arange(
            0, Lq * (B + 1), Lq, dtype=torch.int32, device=q_compact.device
        )
        if q_compact.dtype == torch.float32:
            oup = flash_attn_varlen_kvpacked_func(
                q=q_compact.to(dtype=torch.bfloat16),
                kv=kv_compact.to(dtype=torch.bfloat16),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=Lq,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0,
                softmax_scale=self.scale,
            ).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(
                q=q_compact,
                kv=kv_compact,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=Lq,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0,
                softmax_scale=self.scale,
            ).reshape(B, Lq, -1)
        return self.proj_drop(self.proj(oup))


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        kv_dim,
        cross_attn_layer_scale,
        cond_dim,
        act,
        shared_aln,
        norm_layer,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        tau=1,
        cos_attn=False,
        swiglu=False,
        customized_flash_attn=False,
        fused_mlp=False,
        fused_norm_func=None,
        checkpointing_sa_only=False,
        **_unused,
    ):
        super().__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            proj_drop=drop,
            tau=tau,
            cos_attn=cos_attn,
            customized_flash_attn=customized_flash_attn,
        )
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio / 256) * 256,
            drop=drop,
            fused_mlp=fused_mlp,
        )
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get("eps", 1e-6)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(
                torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5
            )
        else:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = (
                nn.Sequential(nn.SiLU(inplace=False), lin)
                if act
                else nn.Sequential(lin)
            )

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector):
        with torch.amp.autocast("cuda", enabled=False):
            if self.shared_aln:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                    self.ada_gss + cond_BD
                ).unbind(2)
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                    self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
                )
        if self.fused_norm_func is None:
            x = x + self.drop_path(
                self.attn(
                    self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                    attn_bias_or_two_vector=attn_bias_or_two_vector,
                ).mul_(gamma1)
            )
            x = x + self.drop_path(
                self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)
            )
        else:
            x = x + self.drop_path(
                self.attn(
                    self.fused_norm_func(
                        C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1
                    ),
                    attn_bias_or_two_vector=attn_bias_or_two_vector,
                ).mul_(gamma1)
            )
            x = x + self.drop_path(
                self.ffn(
                    self.fused_norm_func(
                        C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2
                    )
                ).mul(gamma2)
            )
        return x


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        kv_dim,
        cross_attn_layer_scale,
        cond_dim,
        act,
        shared_aln,
        norm_layer,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        tau=1,
        cos_attn=False,
        swiglu=False,
        customized_flash_attn=False,
        fused_mlp=False,
        fused_norm_func=None,
        checkpointing_sa_only=False,
        use_flex_attn=False,
        batch_size=2,
        pad_to_multiplier=1,
        apply_rope2d=False,
        rope2d_normalized_by_hw=False,
    ):
        super().__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sa = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            proj_drop=drop,
            tau=tau,
            cos_attn=cos_attn,
            customized_flash_attn=customized_flash_attn,
            use_flex_attn=use_flex_attn,
            batch_size=batch_size,
            pad_to_multiplier=pad_to_multiplier,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        )
        self.ca = CrossAttention(
            embed_dim=embed_dim,
            kv_dim=kv_dim,
            num_heads=num_heads,
            proj_drop=drop,
            cos_attn=cos_attn,
        )
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio / 256) * 256,
            drop=drop,
            fused_mlp=fused_mlp,
        )
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get("eps", 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(
                torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5
            )
        else:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = (
                nn.Sequential(nn.SiLU(inplace=False), lin)
                if act
                else nn.Sequential(lin)
            )
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(
                cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True
            )
        else:
            self.ca_gamma = 1
        self.checkpointing_sa_only = checkpointing_sa_only

    def forward(
        self,
        x,
        cond_BD,
        ca_kv,
        attn_bias_or_two_vector,
        attn_fn=None,
        scale_schedule=None,
        rope2d_freqs_grid=None,
        scale_ind=0,
    ):
        with torch.amp.autocast("cuda", enabled=False):
            if self.shared_aln:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                    self.ada_gss + cond_BD
                ).unbind(2)
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                    self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
                )
        if self.fused_norm_func is None:
            x_sa = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(
                    self.sa,
                    x_sa,
                    attn_bias_or_two_vector,
                    attn_fn,
                    scale_schedule,
                    rope2d_freqs_grid,
                    use_reentrant=False,
                )
            else:
                x_sa = self.sa(
                    x_sa,
                    attn_bias_or_two_vector,
                    attn_fn,
                    scale_schedule,
                    rope2d_freqs_grid,
                    scale_ind=scale_ind,
                )
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).mul_(self.ca_gamma)
            x = x + self.drop_path(
                self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)
            )
        else:
            x_sa = self.fused_norm_func(
                C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1
            )
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(
                    self.sa,
                    x_sa,
                    attn_bias_or_two_vector,
                    attn_fn,
                    scale_schedule,
                    rope2d_freqs_grid,
                    use_reentrant=False,
                )
            else:
                x_sa = self.sa(
                    x_sa,
                    attn_bias_or_two_vector,
                    attn_fn,
                    scale_schedule,
                    rope2d_freqs_grid,
                    scale_ind=scale_ind,
                )
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).mul_(self.ca_gamma)
            x = x + self.drop_path(
                self.ffn(
                    self.fused_norm_func(
                        C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2
                    )
                ).mul(gamma2)
            )
        return x


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, act, norm_layer, fused_norm_func=None):
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get("eps", 1e-6)
        lin = nn.Linear(D, 2 * C)
        self.ada_lin = (
            nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
        )

    def forward(self, x_BLC, cond_BD):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        if self.fused_norm_func is None:
            return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
        return self.fused_norm_func(
            C=self.C, eps=self.norm_eps, x=x_BLC, scale=scale, shift=shift
        )


# =====================================================================
# 9. Infinity transformer (infinity/models/infinity.py)
# =====================================================================
class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5, D):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        self.head_dim = 64 if D > 4096 else 128
        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(
            for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads
        )

    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index + num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(
        self,
        x,
        cond_BD,
        ca_kv,
        attn_bias_or_two_vector,
        attn_fn=None,
        scale_schedule=None,
        checkpointing_full_block=False,
        rope2d_freqs_grid=None,
    ):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = checkpoint(
                    m,
                    h,
                    cond_BD,
                    ca_kv,
                    attn_bias_or_two_vector,
                    attn_fn,
                    scale_schedule,
                    rope2d_freqs_grid,
                    use_reentrant=False,
                )
            else:
                h = m(
                    h,
                    cond_BD,
                    ca_kv,
                    attn_bias_or_two_vector,
                    attn_fn,
                    scale_schedule,
                    rope2d_freqs_grid,
                )
        return h


class Infinity(nn.Module):
    def __init__(
        self,
        vae_local,
        text_channels=0,
        text_maxlen=0,
        selecting_idx=None,
        embed_dim=1024,
        depth=16,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        rms_norm=False,
        shared_aln=False,
        head_aln=True,
        cond_drop_rate=0.1,
        rand_uncond=False,
        cross_attn_layer_scale=-1.0,
        nm0=False,
        tau=1,
        cos_attn=True,
        swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0,
        top_k=0.0,
        customized_flash_attn=False,
        fused_mlp=False,
        fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify=0,
        inference_mode=False,
    ):
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        self.d_vae = (
            vae_local.embed_dim * 4
            if self.apply_spatial_patchify
            else vae_local.embed_dim
        )
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = (
            train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        )
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        assert add_lvl_embeding_only_first_block in [0, 1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0, 1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        self.customized_flash_attn = False
        self.raw_scale_schedule = raw_scale_schedule
        self.first_l = 1
        self.top_p = max(min(top_p, 1), 0)
        self.top_k = round(top_k * self.V) if 0 < top_k < 1 else round(top_k)
        if self.top_p < 1e-5:
            self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0:
            self.top_k = 0

        super().__init__()
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0

        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        if self.t2i:
            self.selecting_idx = None
            self.num_classes = 0
            self.D = self.C
            cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
            rng = torch.Generator(device="cpu")
            rng.manual_seed(0)
            nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
            cfg_uncond /= self.Ct5**0.5
            if rand_uncond:
                self.register_buffer("cfg_uncond", cfg_uncond)
            else:
                self.cfg_uncond = nn.Parameter(cfg_uncond)
            self.text_norm = FastRMSNorm(
                self.Ct5, elementwise_affine=True, eps=norm_eps
            )
            self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.D, self.D),
            )
        else:
            if selecting_idx is None:
                num_classes = 1000
                selecting_idx = torch.full(
                    (1, num_classes),
                    fill_value=1 / num_classes,
                    dtype=torch.float32,
                    device=dist.get_device(),
                )
            self.selecting_idx = selecting_idx
            self.num_classes = selecting_idx.shape[-1]
            self.D = self.C
            self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)

        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        if self.rope2d_each_sa_layer:
            self.rope2d_freqs_grid = precompute_rope2d_freqs_grid(
                dim=self.C // self.num_heads,
                dynamic_resolution_h_w=dynamic_resolution_h_w,
                pad_to_multiplier=self.pad_to_multiplier,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
            )
        else:
            raise ValueError("rope2d_each_sa_layer=0 not supported")
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        self.shared_ada_lin = (
            nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6 * self.C))
            if shared_aln
            else nn.Identity()
        )

        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
        else:
            fused_norm_func = None

        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = (CrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C,
                kv_dim=self.D,
                cross_attn_layer_scale=cross_attn_layer_scale,
                cond_dim=self.D,
                act=True,
                shared_aln=shared_aln,
                norm_layer=norm_layer,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[block_idx],
                tau=tau,
                cos_attn=cos_attn,
                swiglu=swiglu,
                customized_flash_attn=self.customized_flash_attn,
                fused_mlp=fused_mlp,
                fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == "self-attn",
                use_flex_attn=use_flex_attn,
                batch_size=batch_size,
                pad_to_multiplier=pad_to_multiplier,
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)

        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(
                self.C,
                self.D,
                act=True,
                norm_layer=norm_layer,
                fused_norm_func=fused_norm_func,
            )
            self.head = (
                nn.Linear(self.C, V)
                if head_depth == 1
                else nn.Sequential(
                    nn.Linear(self.C, self.C, bias=True),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(self.C, V),
                )
            )
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V))

        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(
                    MultipleLayers(
                        self.unregistered_blocks,
                        self.num_blocks_in_a_chunk,
                        i * self.num_blocks_in_a_chunk,
                    )
                )

    def get_logits(self, h, cond_BD):
        with torch.amp.autocast("cuda", enabled=False):
            return self.head(self.head_nm(h, cond_BD))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(
            scale_ind
            * torch.ones((bs, t_mul_h_mul_w), dtype=torch.int).to(feature.device)
        )
        return feature

    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = int(np.array(patch_t_h_w).prod())
            x_BLC_this = x_BLC[:, ptr : ptr + scale_seq_len]
            ptr += scale_seq_len
            x_BLC_this = self.add_lvl_embeding(x_BLC_this, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this)
        assert x_BLC.shape[1] == (ptr + need_to_pad)
        x_BLC_list.append(x_BLC[:, ptr:])
        return torch.cat(x_BLC_list, dim=1)

    def forward(
        self, label_B_or_BLT, x_BLC_wo_prefix, scale_schedule, cfg_infer=False, **kwargs
    ):
        if cfg_infer:
            return self.autoregressive_infer_cfg(
                label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, **kwargs
            )

        # NOTE: `next(self.parameters()).dtype` breaks TorchDynamo tracing
        # (InternalTorchDynamoError on the parameters() generator). Read the
        # dtype from a concrete always-present parameter instead -- equivalent
        # value, but a plain attribute access that traces cleanly.
        _model_dtype = self.word_embed.weight.dtype
        x_BLC_wo_prefix = x_BLC_wo_prefix.to(_model_dtype)
        B = x_BLC_wo_prefix.shape[0]
        with torch.amp.autocast("cuda", enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total : total + le] = self.cfg_uncond[:le]
                total += le
            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.text_norm(kv_compact).contiguous()
            sos = cond_BD = (
                self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
                .to(_model_dtype)
                .contiguous()
            )
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            x_BLC = torch.cat(
                (sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1
            )
            l_end = x_BLC.shape[1]
            need_to_pad = (
                l_end + self.pad_to_multiplier - 1
            ) // self.pad_to_multiplier * self.pad_to_multiplier - l_end
            if self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = None
            else:
                d = torch.cat(
                    [
                        torch.full((pn[0] * pn[1] * pn[2],), i)
                        for i, pn in enumerate(scale_schedule)
                    ]
                ).view(1, l_end, 1)
                dT = d.transpose(1, 2)
                attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
                    1, 1, l_end, l_end
                )
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()
                if need_to_pad:
                    attn_bias = F.pad(
                        attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf
                    )
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)

        attn_fn = (
            self.attn_fn_compile_dict[tuple(scale_schedule)]
            if self.use_flex_attn
            else None
        )
        checkpointing_full_block = self.checkpointing == "full-block" and self.training
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(
                        x_BLC, scale_schedule, need_to_pad
                    )
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(
                        x_BLC, scale_schedule, need_to_pad
                    )
                if checkpointing_full_block:
                    x_BLC = checkpoint(
                        b,
                        x_BLC,
                        cond_BD_or_gss,
                        ca_kv,
                        attn_bias_or_two_vector,
                        attn_fn,
                        scale_schedule,
                        self.rope2d_freqs_grid,
                        use_reentrant=False,
                    )
                else:
                    x_BLC = b(
                        x=x_BLC,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv,
                        attn_bias_or_two_vector=attn_bias_or_two_vector,
                        attn_fn=attn_fn,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=self.rope2d_freqs_grid,
                    )
        else:
            for i, chunk in enumerate(self.block_chunks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(
                        x_BLC, scale_schedule, need_to_pad
                    )
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(
                        x_BLC, scale_schedule, need_to_pad
                    )
                x_BLC = chunk(
                    x=x_BLC,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=attn_bias_or_two_vector,
                    attn_fn=attn_fn,
                    scale_schedule=scale_schedule,
                    checkpointing_full_block=checkpointing_full_block,
                    rope2d_freqs_grid=self.rope2d_freqs_grid,
                )

        return self.get_logits(x_BLC[:, :l_end], cond_BD)

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1,
        negative_label_B_or_BLT=None,
        force_gt_Bhw=None,
        g_seed=None,
        cfg_list=[],
        tau_list=[],
        cfg_sc=3,
        top_k=0,
        top_p=0.0,
        returns_vemb=0,
        ratio_Bl1=None,
        gumbel=0,
        norm_cfg=False,
        cfg_exp_k=0.0,
        cfg_insertion_layer=[-5],
        vae_type=0,
        softmax_merge_topk=-1,
        ret_img=False,
        trunk_scale=1000,
        gt_leak=0,
        gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
    ):
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        vae_scale_schedule = (
            [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
            if self.apply_spatial_patchify
            else scale_schedule
        )
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2 * B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total : total + le] = self.cfg_uncond[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat(
                    (cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0
                )
            else:
                (
                    kv_compact_un,
                    lens_un,
                    cu_seqlens_k_un,
                    max_seqlen_k_un,
                ) = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat(
                    (cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0
                )
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
        kv_compact = self.text_proj_for_ca(kv_compact)
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(
            bs, 1, -1
        )
        with torch.amp.autocast("cuda", enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []
        idx_Bl_list, idx_Bld_list = [], []
        if inference_mode:
            for b in self.unregistered_blocks:
                (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (
                        module.sa if isinstance(module, CrossAttnBlock) else module.attn
                    ).kv_caching(True)

        abs_cfg_insertion_layers = []
        add_cfg_on_logits = add_cfg_on_probs = False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0:
                add_cfg_on_logits = True
            elif item == 1:
                add_cfg_on_probs = True
            elif item < 0:
                abs_cfg_insertion_layers.append(leng + item)
            else:
                raise ValueError

        num_stages_minus_1 = len(scale_schedule) - 1
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            cur_L += int(np.array(pn).prod())
            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                attn_fn = self.attn_fn_compile_dict.get(
                    tuple(scale_schedule[: (si + 1)]), None
                )
            layer_idx = 0
            for block_idx, b in enumerate(self.block_chunks):
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(
                        last_stage, si, scale_schedule, need_to_pad=need_to_pad
                    )
                if not self.add_lvl_embeding_only_first_block:
                    last_stage = self.add_lvl_embeding(
                        last_stage, si, scale_schedule, need_to_pad=need_to_pad
                    )
                for m in b.module:
                    last_stage = m(
                        x=last_stage,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv,
                        attn_bias_or_two_vector=None,
                        attn_fn=attn_fn,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=self.rope2d_freqs_grid,
                        scale_ind=si,
                    )
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        last_stage = cfg * last_stage[:B] + (1 - cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1

            if (cfg != 1) and add_cfg_on_logits:
                logits_BlV = self.get_logits(last_stage, cond_BD).mul(1 / tau_list[si])
                logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(
                    1 / tau_list[si]
                )

            if self.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits_BlV,
                    rng=rng,
                    top_k=top_k or self.top_k,
                    top_p=top_p or self.top_p,
                    num_samples=1,
                )[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits_BlV,
                    rng=rng,
                    top_k=top_k or self.top_k,
                    top_p=top_p or self.top_p,
                    num_samples=1,
                )[:, :, 0]
            if vae_type != 0:
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
                    if self.apply_spatial_patchify:
                        idx_Bld = idx_Bld.permute(0, 3, 1, 2)
                        idx_Bld = F.pixel_shuffle(idx_Bld, 2)
                        idx_Bld = idx_Bld.permute(0, 2, 3, 1)
                    idx_Bld = idx_Bld.unsqueeze(1)
                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(
                    idx_Bld, label_type="bit_label"
                )
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(
                        codes,
                        size=vae_scale_schedule[-1],
                        mode=vae.quantizer.z_interplote_up,
                    )
                    last_stage = F.interpolate(
                        summed_codes,
                        size=vae_scale_schedule[si + 1],
                        mode=vae.quantizer.z_interplote_up,
                    )
                    last_stage = last_stage.squeeze(-3)
                    if self.apply_spatial_patchify:
                        last_stage = F.pixel_unshuffle(last_stage, 2)
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
                    last_stage = torch.permute(last_stage, [0, 2, 1])
                else:
                    summed_codes += codes

            if si != num_stages_minus_1:
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs // B, 1, 1)

        if inference_mode:
            for b in self.unregistered_blocks:
                (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (
                        module.sa if isinstance(module, CrossAttnBlock) else module.attn
                    ).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        img = vae.decode(summed_codes.squeeze(-3))
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img

    def load_state_dict(self, state_dict, strict=False, assign=False):
        for k in state_dict:
            if "cfg_uncond" in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat(
                        (old.to(device=new.device, dtype=new.dtype), new[min_tlen:])
                    )
                else:
                    state_dict[k] = old[:min_tlen]
        for buf_name in (
            "lvl_1L",
            "attn_bias_for_masking",
            "Infinity_visible_kvlen",
            "Infinity_invisible_qlen",
        ):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        return super().load_state_dict(
            state_dict=state_dict, strict=strict, assign=assign
        )


def sample_with_top_k_top_p_also_inplace_modifying_logits_(
    logits_BlV, top_k=0, top_p=0.0, rng=None, num_samples=1
):
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (
            1 - top_p
        )
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(
                sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
            ),
            -torch.inf,
        )
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng,
    ).view(B, l, num_samples)


# =====================================================================
# 10. Loader helpers (tools/run_infinity.py — minimal subset)
# =====================================================================
def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    captions = [prompt]
    tokens = text_tokenizer(
        text=captions,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc_device = next(text_encoder.parameters()).device
    input_ids = tokens.input_ids.to(enc_device, non_blocking=True)
    mask = tokens.attention_mask.to(enc_device, non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)[
        "last_hidden_state"
    ].float()
    lens = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    return kv_compact, lens, cu_seqlens_k, Ltext


def load_tokenizer(t5_path):
    """Load the T5-XL tokenizer + encoder from a local directory or HF id."""
    from transformers import AutoTokenizer, T5EncoderModel

    text_tokenizer = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    text_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder


def load_visual_tokenizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert args.vae_type in [14, 16, 18, 20, 24, 32, 64]
    schedule_mode = "dynamic"
    codebook_dim = args.vae_type
    codebook_size = 2**codebook_dim
    if args.apply_spatial_patchify:
        patch_size = 8
        encoder_ch_mult = [1, 2, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4]
    else:
        patch_size = 16
        encoder_ch_mult = [1, 2, 4, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4, 4]
    return vae_model(
        args.vae_path,
        schedule_mode,
        codebook_dim,
        codebook_size,
        patch_size=patch_size,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        test_mode=True,
    ).to(device)


def load_infinity(
    rope2d_each_sa_layer,
    rope2d_normalized_by_hw,
    use_scale_schedule_embedding,
    pn,
    use_bit_label,
    add_lvl_embeding_only_first_block,
    model_path="",
    scale_schedule=None,
    vae=None,
    device="cuda",
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type="torch",
):
    text_maxlen = 512
    _amp_dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.amp.autocast(
        _amp_dev,
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
        cache_enabled=True,
    ), torch.no_grad():
        infinity_test = Infinity(
            vae_local=vae,
            text_channels=text_channels,
            text_maxlen=text_maxlen,
            shared_aln=True,
            raw_scale_schedule=scale_schedule,
            checkpointing="full-block",
            customized_flash_attn=False,
            fused_norm=False,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()
        infinity_test.eval()
        infinity_test.requires_grad_(False)
        if torch.cuda.is_available():
            infinity_test.cuda()
            torch.cuda.empty_cache()
        if checkpoint_type == "torch":
            state_dict = torch.load(model_path, map_location=device)
            infinity_test.load_state_dict(state_dict)
    return infinity_test


def load_transformer(vae, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path
    kwargs_model = dict(
        depth=32,
        embed_dim=2048,
        num_heads=2048 // 128,
        drop_path_rate=0.1,
        mlp_ratio=4,
        block_chunks=8,
    )
    return load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        model_path=model_path,
        scale_schedule=None,
        vae=vae,
        device=device,
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=bool(args.use_flex_attn),
        bf16=bool(args.bf16),
        checkpoint_type=args.checkpoint_type,
    )
