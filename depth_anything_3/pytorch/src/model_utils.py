# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from einops import rearrange


def _position_getter_no_cache(self, batch_size, height, width, device):
    """Device-aware position getter that bypasses the cache."""
    y_coords = torch.arange(height, device=device)
    x_coords = torch.arange(width, device=device)
    positions = torch.cartesian_prod(y_coords, x_coords)
    return positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


def _prepare_rope_xla_compat(self, B, S, H, W, device):
    """XLA-compatible _prepare_rope that creates tensors directly on device."""
    pos = None
    pos_nodiff = None
    if self.rope is not None:
        pos = self.position_getter(
            B * S, H // self.patch_size, W // self.patch_size, device=device
        )
        pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
        pos_nodiff = torch.zeros_like(pos)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype
            )
            pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
            pos = torch.cat([pos_special, pos], dim=2)
            pos_nodiff = pos_nodiff + 1
            pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
    return pos, pos_nodiff


def patch_da3_for_xla(model):
    """Patch DA3 model for XLA compatibility."""
    import types

    backbone = model.backbone.pretrained
    backbone._prepare_rope = types.MethodType(_prepare_rope_xla_compat, backbone)
    if backbone.position_getter is not None:
        backbone.position_getter.__call__ = types.MethodType(
            _position_getter_no_cache, backbone.position_getter
        )
