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


def _channel_sliced_conv1d_forward(self, hidden_states):
    # Depthwise, so channels are independent: no halo, no partial sums, and the
    # result is identical to the unpatched module, padding included.
    chunk = self._tt_conv_channel_chunk
    outputs = []
    for start in range(0, self.in_channels, chunk):
        end = min(start + chunk, self.in_channels)
        outputs.append(
            F.conv1d(
                hidden_states[:, start:end, :],
                self.weight[start:end],
                self.bias[start:end] if self.bias is not None else None,
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
    patched = 0
    for module in model.modules():
        if not _is_depthwise_conv1d(module) or module.in_channels <= chunk:
            continue
        module._tt_conv_channel_chunk = chunk
        module.forward = _channel_sliced_conv1d_forward.__get__(module, type(module))
        patched += 1
    return patched
