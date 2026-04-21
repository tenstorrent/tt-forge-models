# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for CatVTON (zhengchong/CatVTON) virtual try-on model loading
and preprocessing. CatVTON wraps a Stable Diffusion v1.5 inpainting pipeline,
swaps the VAE with stabilityai/sd-vae-ft-mse, replaces every cross-attention
processor with a no-op SkipAttnProcessor, and then loads self-attention
weights from the zhengchong/CatVTON repository.
"""

import os

import torch
from accelerate import load_checkpoint_in_model
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw


CATVTON_REPO = "zhengchong/CatVTON"
CATVTON_VAE = "stabilityai/sd-vae-ft-mse"


class SkipAttnProcessor(torch.nn.Module):
    """No-op cross-attention processor used by CatVTON to bypass text conditioning."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        return hidden_states


def _skip_cross_attention(unet):
    """Replace every cross-attention processor in the UNet with SkipAttnProcessor."""
    attn_procs = {}
    for name, processor in unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            attn_procs[name] = processor
        else:
            attn_procs[name] = SkipAttnProcessor()
    unet.set_attn_processor(attn_procs)


def _load_catvton_attention_weights(unet, attn_ckpt_version):
    """Download and load CatVTON self-attention weights into the UNet."""
    snapshot_dir = snapshot_download(
        repo_id=CATVTON_REPO,
        allow_patterns=[f"{attn_ckpt_version}/attention/*"],
    )
    attention_dir = os.path.join(snapshot_dir, attn_ckpt_version, "attention")

    self_attn_modules = torch.nn.ModuleList(
        [module for name, module in unet.named_modules() if name.endswith("attn1")]
    )
    load_checkpoint_in_model(self_attn_modules, attention_dir)


def load_catvton_pipe(base_model_name, attn_ckpt_version):
    """Load the CatVTON virtual try-on pipeline.

    Args:
        base_model_name: HuggingFace name of the SD 1.5 inpainting base checkpoint.
        attn_ckpt_version: CatVTON attention checkpoint subdirectory
            (e.g. "mix-48k-1024", "vitonhd-16k-512", "dresscode-16k-512").

    Returns:
        StableDiffusionInpaintPipeline: Pipeline with CatVTON VAE, attention
            processors and attention weights applied, in eval mode on CPU.
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_name, torch_dtype=torch.float32, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderKL.from_pretrained(CATVTON_VAE)

    _skip_cross_attention(pipe.unet)
    _load_catvton_attention_weights(pipe.unet, attn_ckpt_version)

    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.unet, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def create_dummy_input_image(height=512, width=512):
    """Create a dummy person image for virtual try-on."""
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def create_dummy_mask_image(height=512, width=512):
    """Create a dummy garment-agnostic mask (white region to inpaint)."""
    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)
    center_x, center_y = width // 2, height // 2
    box_size = min(width, height) // 4
    draw.rectangle(
        [
            center_x - box_size,
            center_y - box_size,
            center_x + box_size,
            center_y + box_size,
        ],
        fill=255,
    )
    return mask


def catvton_preprocessing(
    pipe,
    prompt,
    image,
    mask_image,
    device="cpu",
    num_inference_steps=10,
    guidance_scale=7.5,
    num_images_per_prompt=1,
):
    """Produce UNet-ready tensors for a CatVTON forward pass.

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds)
    """
    height = width = pipe.unet.config.sample_size * pipe.vae_scale_factor

    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    image = pipe.image_processor.preprocess(image, height=height, width=width)
    mask = pipe.mask_processor.preprocess(mask_image, height=height, width=width)

    masked_image = image * (mask < 0.5)
    masked_image_latents = (
        pipe.vae.encode(masked_image).latent_dist.mode()
        * pipe.vae.config.scaling_factor
    )

    mask = torch.nn.functional.interpolate(
        mask,
        size=(height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
    )

    if do_classifier_free_guidance:
        masked_image_latents = torch.cat([masked_image_latents] * 2)
        mask = torch.cat([mask] * 2)

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    num_channels_latents = (
        pipe.unet.config.in_channels - mask.shape[1] - masked_image_latents.shape[1]
    )
    torch.manual_seed(42)
    latents = torch.randn(
        (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    scaled_latent_model_input = pipe.scheduler.scale_model_input(latents, timesteps[0])
    if do_classifier_free_guidance:
        scaled_latent_model_input = torch.cat([scaled_latent_model_input] * 2)

    latent_model_input = torch.cat(
        [scaled_latent_model_input, mask, masked_image_latents], dim=1
    )

    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds
