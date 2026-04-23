# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for Stable Diffusion v1.5 (stablediffusiontutorials) model loading."""

import torch
from diffusers import StableDiffusionPipeline


def load_pipe(pretrained_model_name):
    """Load Stable Diffusion v1.5 pipeline.

    Args:
        pretrained_model_name: HuggingFace repo ID

    Returns:
        StableDiffusionPipeline: Loaded pipeline with UNet in eval mode
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.bfloat16
    )
    pipe.unet.eval()
    for param in pipe.unet.parameters():
        param.requires_grad_(False)
    return pipe


def stable_diffusion_preprocessing(pipe, prompt, device="cpu", num_inference_steps=50):
    """Preprocess inputs for SD v1.5 UNet.

    Args:
        pipe: StableDiffusionPipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 50)

    Returns:
        dict: UNet keyword arguments with keys:
            - sample: Latent tensor (1, 4, 64, 64)
            - timestep: Timestep tensor
            - encoder_hidden_states: Text embeddings (1, 77, 768)
    """
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timestep = pipe.scheduler.timesteps[0]

    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]

    latent = torch.randn(1, pipe.unet.config.in_channels, 64, 64, dtype=torch.bfloat16)

    return {
        "sample": latent,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
    }
