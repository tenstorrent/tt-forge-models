# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Convolution rewrites for shapes the TT backend cannot run as-is."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_depthwise_conv1d(module) -> bool:
    if not isinstance(module, nn.Conv1d):
        return False
    return (
        module.groups == module.in_channels
        and module.in_channels == module.out_channels
    )


class ChannelSlicedDepthwiseConv1d(nn.Module):
    """User module so Dynamo traces the channel split instead of ``nn.Conv1d``.

    Patching ``Conv1d.forward`` is not enough under torch.compile: Dynamo still
    inlines the builtin ``Conv1d`` into a single ``aten.convolution``. Replacing
    the module with this class makes the sliced ``F.conv1d`` calls show up in
    the graph.
    """

    def __init__(self, conv: nn.Conv1d, chunk: int):
        super().__init__()
        # Keep the same Parameter objects so load_shard_spec keys still match.
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.groups = conv.groups
        self.kernel_size = conv.kernel_size
        self._tt_conv_channel_chunk = chunk

    def forward(self, hidden_states):
        # Depthwise, so channels are independent: no halo, no partial sums, and
        # the result is identical to the unpatched module, padding included.
        chunk = self._tt_conv_channel_chunk
        outputs = []
        for start in range(0, self.in_channels, chunk):
            end = min(start + chunk, self.in_channels)
            bias = None if self.bias is None else self.bias[start:end]
            outputs.append(
                F.conv1d(
                    hidden_states[:, start:end, :],
                    self.weight[start:end],
                    bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=end - start,
                )
            )
        return torch.cat(outputs, dim=1)


def slice_depthwise_conv1d_channels(model, chunk: int) -> int:
    """Split wide depthwise conv1ds in ``model`` into ``chunk``-channel slices.

    ttnn can only slice a conv on height and width, so a deep but spatially tiny
    one has no usable axis and aborts. Channels are the free axis here.

    ``chunk`` has no default because the workable size depends on free L1, which
    varies by arch and mesh; each loader passes its own measured value. Use a
    divisor of the channel count -- 10240 with chunk 4096 leaves a 2048 tail,
    and 2048 has been measured not to fit.

    Returns the number of modules patched.
    """
    to_replace = []
    for name, module in model.named_modules():
        if not _is_depthwise_conv1d(module) or module.in_channels <= chunk:
            continue
        to_replace.append(name)

    for name in to_replace:
        parent = model
        *parents, attr = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        conv = getattr(parent, attr)
        setattr(parent, attr, ChannelSlicedDepthwiseConv1d(conv, chunk))
    return len(to_replace)
