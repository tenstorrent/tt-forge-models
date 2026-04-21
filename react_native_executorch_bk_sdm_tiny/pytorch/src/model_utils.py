# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for BK-SDM-Tiny model loading and preprocessing.
"""

import torch
from diffusers import StableDiffusionPipeline


def load_pipe(model_name: str) -> StableDiffusionPipeline:
    """Load BK-SDM-Tiny pipeline and set components to eval mode.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        StableDiffusionPipeline with components set to eval mode.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    pipe.to("cpu")
    for module in [pipe.text_encoder, pipe.unet, pipe.vae]:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
    return pipe


def stable_diffusion_preprocessing(
    pipe: StableDiffusionPipeline,
    prompt: str,
    device: str = "cpu",
    num_inference_steps: int = 30,
):
    """Prepare UNet inputs for a single forward pass.

    Args:
        pipe: Loaded StableDiffusionPipeline.
        prompt: Text prompt used to generate encoder hidden states.
        device: Device string (default: "cpu").
        num_inference_steps: Number of scheduler timesteps to set.

    Returns:
        tuple: (latents, timestep, encoder_hidden_states)
    """
    # Encode prompt with classifier-free guidance (uncond + cond)
    prompt_embeds, negative_prompt_embeds, *_ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt="",
        device=device,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
    )
    encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # Prepare latents
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = height
    torch.manual_seed(42)
    latents = torch.randn(
        (
            1,
            pipe.unet.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    # Duplicate for classifier-free guidance
    latents = torch.cat([latents] * 2, dim=0)

    # Get timestep
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timestep = pipe.scheduler.timesteps[0].expand(latents.shape[0])

    return latents, timestep, encoder_hidden_states
