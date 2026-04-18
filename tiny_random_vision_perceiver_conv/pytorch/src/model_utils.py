# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from transformers.models.perceiver import modeling_perceiver


def _patched_build_network_inputs(self, inputs, network_input_is_1d=True, interpolate_pos_encoding=False):
    batch_size = inputs.shape[0]
    input_size = inputs.shape[1:3]
    index_dims = inputs.shape[1:-1]

    if len(inputs.shape) > 3 and network_input_is_1d:
        inputs = inputs.reshape(batch_size, -1, inputs.shape[-1])

    if self.position_encoding_type == "trainable":
        pos_enc = self.position_embeddings(
            batch_size, interpolate_pos_encoding, input_size
        )
    elif self.position_encoding_type == "fourier":
        pos_enc = self.position_embeddings(
            index_dims, batch_size, device=inputs.device, dtype=inputs.dtype
        )

    pos_enc = self.positions_projection(pos_enc)

    if not network_input_is_1d:
        sh = inputs.shape
        pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])
    if self.concat_or_add_pos == "concat":
        inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
    elif self.concat_or_add_pos == "add":
        inputs_with_pos = inputs + pos_enc
    return inputs_with_pos, inputs


def _patched_check_or_build_spatial_positions(pos, index_dims, batch_size):
    if pos is None:
        pos = modeling_perceiver.build_linear_positions(index_dims)
        pos = pos[None].expand((batch_size,) + pos.shape)
        pos = pos.reshape(batch_size, -1, pos.shape[-1])
    else:
        if pos.shape[-1] != len(index_dims):
            raise ValueError(
                "Spatial features have the wrong number of dimensions."
            )
    return pos


def _patched_generate_fourier_features(
    pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False
):
    batch_size = pos.shape[0]

    min_freq = 1.0
    freq_bands = torch.stack(
        [
            torch.linspace(start=min_freq, end=res / 2, steps=num_bands)
            for res in max_resolution
        ],
        dim=0,
    )

    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = per_pos_features.flatten(1)

    if sine_only:
        per_pos_features = torch.sin(math.pi * per_pos_features)
    else:
        per_pos_features = torch.cat(
            [
                torch.sin(math.pi * per_pos_features),
                torch.cos(math.pi * per_pos_features),
            ],
            dim=-1,
        )
    if concat_pos:
        per_pos_features = torch.cat(
            [pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1
        )
    return per_pos_features


def patch_perceiver_for_dynamo():
    modeling_perceiver.PerceiverImagePreprocessor._build_network_inputs = (
        _patched_build_network_inputs
    )
    modeling_perceiver._check_or_build_spatial_positions = (
        _patched_check_or_build_spatial_positions
    )
    modeling_perceiver.generate_fourier_features = (
        _patched_generate_fourier_features
    )
