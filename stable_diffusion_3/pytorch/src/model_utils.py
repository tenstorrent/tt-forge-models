# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion v3 model loading and preprocessing.

SD3 and SD3.5 share the StableDiffusion3Pipeline class from diffusers, but
they are released under different repositories and have different bringup
characteristics, so we keep their loaders isolated.
"""

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(
    pretrained_model_name: str, dtype: torch.dtype = torch.float32
) -> StableDiffusion3Pipeline:
    """Load a Stable Diffusion v3 pipeline.

    Args:
        pretrained_model_name: The HuggingFace repo name (under ``stabilityai/``).
        dtype: ``torch_dtype`` passed to ``from_pretrained`` so the weights are
            materialized directly at the target dtype. SD3-medium is ~30 GB in
            fp32 vs ~15 GB in bf16; loading fp32 first and casting afterwards
            OOM-kills a 31 GB host, so callers should pass ``torch.bfloat16``.

    Returns:
        StableDiffusion3Pipeline: Loaded pipeline with all sub-modules set to
        eval mode and requires_grad disabled.
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        f"stabilityai/{pretrained_model_name}", torch_dtype=dtype
    )

    pipe.to("cpu")

    modules = [
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
        pipe.transformer,
        pipe.vae,
    ]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def calculate_shift(
    image_seq_len,
    base_image_seq_len,
    max_image_seq_len,
    base_shift,
    max_shift,
):
    """Calculate the dynamic shifting parameter ``mu`` for the SD3 scheduler."""
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return image_seq_len * m + b


def stable_diffusion_preprocessing_v3(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.0,
    num_inference_steps=1,
    num_images_per_prompt=1,
    clip_skip=None,
    max_sequence_length=256,
    do_classifier_free_guidance=True,
    mu=None,
):
    """Run the SD3 pipeline preprocessing (encode_prompt, latents, timestep).

    This mirrors :func:`stable_diffusion_preprocessing_v35` but is kept in this
    module so SD3 has no runtime dependency on the SD3.5 loader package.

    Returns:
        tuple: ``(latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds)``
        — the four tensors expected by ``SD3Transformer2DModel.forward``.
    """
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor

    pipe.check_inputs(
        prompt,
        None,  # prompt_2
        None,  # prompt_3
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        device=device,
        clip_skip=clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

    num_channels_latents = pipe.transformer.config.in_channels
    shape = (
        num_images_per_prompt,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)

    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        # Use the latent spatial dims (not the original image height/width) to
        # match the reference SD3 pipeline implementation.
        _, _, latent_height, latent_width = latents.shape
        image_seq_len = (latent_height // pipe.transformer.config.patch_size) * (
            latent_width // pipe.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=None,
        **scheduler_kwargs,
    )

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds
