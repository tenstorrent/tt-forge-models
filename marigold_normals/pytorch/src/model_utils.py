# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Marigold Normals model loading and preprocessing.
"""

import torch
from diffusers import MarigoldNormalsPipeline


def load_pipe(variant, dtype=None):
    dtype = dtype or torch.float32
    pipe = MarigoldNormalsPipeline.from_pretrained(variant, torch_dtype=dtype)
    pipe.to("cpu")

    for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    return pipe


def marigold_normals_preprocessing(pipe, num_inference_steps=10):
    pipe.scheduler.set_timesteps(num_inference_steps, device="cpu")
    timesteps = pipe.scheduler.timesteps
    timestep = timesteps[0]

    text_inputs = pipe.tokenizer(
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]

    num_channels = pipe.unet.config.in_channels
    height = pipe.unet.config.sample_size
    width = pipe.unet.config.sample_size

    torch.manual_seed(42)
    sample = torch.randn(1, num_channels, height, width)

    return sample, timestep, encoder_hidden_states
