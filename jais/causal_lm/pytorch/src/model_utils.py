# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
JAIS model utilities: patches transformers to restore APIs removed in
transformers 5.x but needed by the JAIS custom modeling code:
- find_pruneable_heads_and_indices and prune_conv1d_layer in pytorch_utils
- model_parallel_utils module (assert_device_map, get_device_map)
"""

import sys
import types

import torch
import transformers.pytorch_utils as _pu
import transformers.utils


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """Find the heads and their indices that need to be pruned."""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_conv1d_layer(layer, index, dim=1):
    """Prune a Conv1D layer to keep only entries at the given index."""
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = _pu.Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

if not hasattr(_pu, "prune_conv1d_layer"):
    _pu.prune_conv1d_layer = prune_conv1d_layer


# Restore transformers.utils.model_parallel_utils removed in transformers 5.x
def assert_device_map(device_map, num_blocks):
    """Validate that a device map covers all blocks."""
    blocks = list(range(num_blocks))
    device_map_blocks = sorted(x for sub in device_map.values() for x in sub)
    assert device_map_blocks == blocks, f"Device map must cover all {num_blocks} blocks"


def get_device_map(num_blocks, devices):
    """Distribute blocks evenly across devices."""
    per_device = (num_blocks + len(devices) - 1) // len(devices)
    device_map = {}
    for i, dev in enumerate(devices):
        device_map[dev] = list(
            range(i * per_device, min((i + 1) * per_device, num_blocks))
        )
    return device_map


if "transformers.utils.model_parallel_utils" not in sys.modules:
    _mpu = types.ModuleType("transformers.utils.model_parallel_utils")
    _mpu.assert_device_map = assert_device_map
    _mpu.get_device_map = get_device_map
    sys.modules["transformers.utils.model_parallel_utils"] = _mpu
    transformers.utils.model_parallel_utils = _mpu


# Patch PretrainedConfig to restore add_cross_attention default removed in transformers 5.x
from transformers import PretrainedConfig

_orig_config_init = PretrainedConfig.__init__


def _patched_config_init(self, **kwargs):
    _orig_config_init(self, **kwargs)
    if not hasattr(self, "add_cross_attention"):
        self.add_cross_attention = False


PretrainedConfig.__init__ = _patched_config_init
