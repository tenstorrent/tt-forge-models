# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VAE decoder op rewrites needed to fit the full-res decoder in device DRAM.

Both rewrites are value-preserving restatements of stock diffusers ops, aimed
at intermediates that dominate the decoder's DRAM footprint. Neither changes
what the decoder computes; see the per-function docstrings for why the stock
form is expensive. Applied together with the ``up_block.proj`` shard spec in
``utils.shard_vae_decoder_specs``, they take the decoder's peak tensor from
42.3 GB to 1.67 GB.

Call :func:`patch_vae_decoder_ops` before the first forward.
"""

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
from diffusers.models.autoencoders.autoencoder_kl_mochi import MochiChunkedGroupNorm3D

_STOCK_FAKE_CONTEXT_PARALLEL_FORWARD = (
    CogVideoXCausalConv3d.fake_context_parallel_forward
)


def _replicate_pad_in_native_dtype(self, inputs, conv_cache=None):
    """CogVideoXCausalConv3d replicate padding without the f32 round trip.

    F.pad(..., mode="replicate") preserves bf16 in eager torch but not in the
    torch-xla lowering: the incoming graph carries convert(bf16 -> f32)
    immediately before the three replicate-pad gathers and convert(f32 -> bf16)
    immediately after - an adjacent mutually-inverse pair that neither XLA nor
    tt-mlir folds. The permute and all three pad concats therefore run at f32,
    each twice the DRAM it needs, on the largest tensors in the decoder.

    Replication is clamp(index) per axis and therefore separable, so padding
    one axis at a time with slice + expand + cat gives identical values while
    staying in the input's own dtype. Order across axes doesn't matter.

    Verified against F.pad by
    tests/torch/models/mochi/check_replicate_pad_equivalence.py.
    """
    if self.pad_mode != "replicate":
        return _STOCK_FAKE_CONTEXT_PARALLEL_FORWARD(self, inputs, conv_cache)

    x = inputs
    if self.time_pad:
        # Causal: leading pad only, replicating frame 0.
        x = torch.cat([x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1), x], dim=2)
    if self.height_pad:
        pad = self.height_pad
        x = torch.cat(
            [
                x[:, :, :, :1].expand(-1, -1, -1, pad, -1),
                x,
                x[:, :, :, -1:].expand(-1, -1, -1, pad, -1),
            ],
            dim=3,
        )
    if self.width_pad:
        pad = self.width_pad
        x = torch.cat(
            [
                x[..., :1].expand(-1, -1, -1, -1, pad),
                x,
                x[..., -1:].expand(-1, -1, -1, -1, pad),
            ],
            dim=4,
        )
    return x


def _group_norm_with_affine_on_channels(self, x: torch.Tensor = None) -> torch.Tensor:
    """MochiChunkedGroupNorm3D with the affine outside F.group_norm.

    F.group_norm normalizes in group space, [chunks*groups, C/groups*H*W],
    where a per-channel weight varies along the *inner* axis and so cannot
    broadcast implicitly. Once up_block.proj is sharded the compiler keeps the
    affine in that space, and each norm materializes its weight and bias as
    full 1x8x128x480x848 f32 tensors - 1.70 GB apiece, via ttnn.repeat.
    (Pre-sharding runs contain none of these, with or without the DRAM
    space-saving pass, so this is a consequence of the shard spec.)

    Applying the affine ourselves on the [chunk, C, H, W] output puts channel
    back on a real dimension, where the broadcast is implicit and free. Same
    arithmetic: group_norm scales by weight[c] and shifts by bias[c] after
    normalizing, which is exactly what this does.

    The normalize is also open-coded so it can run in the input's dtype.
    F.group_norm upcasts everything to f32, which costs 1.67 GB per
    1x256x1628160 intermediate. Only the reductions actually need f32 - their
    outputs are tiny - so mean and variance are computed in f32 and the
    full-size centre-and-scale runs in bf16. That is a real precision
    reduction: the centred values are O(1) so bf16 carries them fine, but the
    subtraction rounds the mean to bf16 first, which costs accuracy in
    proportion to mean/std.

    Verified against the stock module by
    tests/torch/models/mochi/check_group_norm_affine_equivalence.py.
    """
    batch_size = x.size(0)
    norm = self.norm_layer

    x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)

    normalized_chunks = []
    for chunk in x.split(self.chunk_size, dim=0):
        grouped = chunk.reshape(chunk.shape[0], norm.num_groups, -1)

        stats = grouped.float()
        mean = stats.mean(dim=2, keepdim=True)
        # Biased variance, matching F.group_norm.
        var = stats.var(dim=2, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + norm.eps)

        centred = grouped - mean.to(grouped.dtype)
        normalized_chunks.append((centred * rstd.to(grouped.dtype)).reshape_as(chunk))

    output = torch.cat(normalized_chunks, dim=0)
    if norm.affine:
        output = output * norm.weight.view(1, -1, 1, 1) + norm.bias.view(1, -1, 1, 1)

    return output.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)


def patch_vae_decoder_ops() -> None:
    """Rebind the two decoder ops process-wide, including any CPU reference
    model in the same process. Call before the first forward.

    Process-wide is deliberate - the CPU golden has to run the same math for a
    PCC comparison to mean anything - but it does reach every user of these two
    classes in the process, CogVideoX's VAE included.
    """
    CogVideoXCausalConv3d.fake_context_parallel_forward = _replicate_pad_in_native_dtype
    MochiChunkedGroupNorm3D.forward = _group_norm_with_affine_on_channels
