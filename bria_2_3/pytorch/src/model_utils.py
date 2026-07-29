# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for BRIA 2.3 loading and preprocessing.

BRIA 2.3 (briaai/BRIA-2.3) is an SDXL-class text-to-image model: it reuses
the StableDiffusionXLPipeline, the SDXL UNet, and the SDXL preprocessing
path. We keep BRIA in its own loader package (rather than as a variant of
``stable_diffusion_xl``) so its dependencies and bringup state are isolated.
"""

from typing import Optional, Tuple

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(pretrained_model_name: str) -> DiffusionPipeline:
    """Load the BRIA 2.3 (SDXL-class) pipeline.

    Args:
        pretrained_model_name: HuggingFace repo id, e.g. ``"briaai/BRIA-2.3"``.

    Returns:
        DiffusionPipeline: pipeline with all components on CPU, ``eval()`` mode,
        ``requires_grad`` disabled, and BRIA-specific
        ``force_zeros_for_empty_prompt = False`` applied (required per the
        BRIA 2.3 model card).
    """
    pipe = DiffusionPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float32
    )

    # Required by BRIA 2.3: the empty prompt must not be zeroed out.
    pipe.force_zeros_for_empty_prompt = False

    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def bria_2_3_preprocessing(
    pipe,
    prompt: str,
    device: str = "cpu",
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    num_images_per_prompt: int = 1,
    clip_skip: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
):
    """Run BRIA 2.3 (SDXL-class) preprocessing.

    Mirrors :func:`stable_diffusion_xl.stable_diffusion_preprocessing_xl` but
    kept local to the BRIA loader package so it has no runtime dependency on
    the ``stable_diffusion_xl`` loader.

    Returns:
        tuple: ``(latent_model_input, timesteps, prompt_embeds, added_cond_kwargs)``
        — the positional inputs for the wrapped SDXL UNet.
    """
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    original_size = (height, width)
    target_size = (height, width)

    pipe.check_inputs(
        prompt,
        None,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_steps=None,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    do_classifier_free_guidance = True
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
    )

    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    num_channels_latents = pipe.unet.config.in_channels
    torch.manual_seed(42)
    latents = torch.randn(
        (
            num_images_per_prompt,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    add_time_ids = add_time_ids.repeat(num_images_per_prompt, 1)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )

    return latent_model_input, timesteps, prompt_embeds, added_cond_kwargs


class Bria23UNetWrapper(torch.nn.Module):
    """Wrap the SDXL UNet so it can be driven by positional tensor inputs only.

    The raw SDXL UNet forward takes ``added_cond_kwargs`` (a ``dict``) and a
    scalar ``timestep``. The auto-runner ``DynamicTorchModelTester`` calls
    ``model(*inputs)`` positionally, so we capture the ``added_cond_kwargs``
    dict at construction time and expose a clean
    ``(latent_model_input, timesteps, prompt_embeds)`` interface.
    """

    def __init__(self, unet: torch.nn.Module, added_cond_kwargs: dict) -> None:
        super().__init__()
        self.unet = unet
        self.added_cond_kwargs = added_cond_kwargs

    def forward(
        self,
        latent_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        device = latent_model_input.device
        prompt_embeds = prompt_embeds.to(device)
        # SDXL UNet expects a scalar / 0-D timestep per call.
        timestep_on_device = timesteps[0].to(device)
        added_cond_kwargs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in self.added_cond_kwargs.items()
        }

        noise_pred = self.unet(
            latent_model_input,
            timestep_on_device,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
        )[0]
        return noise_pred
