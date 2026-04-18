# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion 3.5 FP8 model loading and processing.
"""

import json
import os
import tempfile

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, CLIPTextConfig, CLIPTokenizer, T5TokenizerFast

REPO_ID = "Comfy-Org/stable-diffusion-3.5-fp8"

_LARGE_TRANSFORMER_CONFIG = {
    "_class_name": "SD3Transformer2DModel",
    "_diffusers_version": "0.37.1",
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 38,
    "attention_head_dim": 64,
    "num_attention_heads": 38,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 2432,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "pos_embed_max_size": 192,
    "dual_attention_layers": [],
    "qk_norm": "rms_norm",
}

_MEDIUM_TRANSFORMER_CONFIG = {
    "_class_name": "SD3Transformer2DModel",
    "_diffusers_version": "0.37.1",
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 24,
    "attention_head_dim": 64,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 1536,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "pos_embed_max_size": 384,
    "dual_attention_layers": list(range(13)),
    "qk_norm": "rms_norm",
}

_VAE_CONFIG = {
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.37.1",
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "latent_channels": 16,
    "norm_num_groups": 32,
    "act_fn": "silu",
    "scaling_factor": 1.5305,
    "shift_factor": 0.0609,
    "sample_size": 1024,
    "force_upcast": True,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
}

_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.37.1",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


def _build_config_dir(transformer_config):
    """Build a local config directory for StableDiffusion3Pipeline.from_single_file.

    The upstream stabilityai SD3.5 repos are gated, so we construct the config
    locally using non-gated component repos for tokenizers and text encoder configs.
    """
    config_dir = tempfile.mkdtemp()

    with open(os.path.join(config_dir, "model_index.json"), "w") as f:
        json.dump(
            {
                "_class_name": "StableDiffusion3Pipeline",
                "_diffusers_version": "0.37.1",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
                "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
                "text_encoder_3": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "CLIPTokenizer"],
                "tokenizer_3": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "SD3Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            },
            f,
            indent=2,
        )

    for name, cfg in [
        ("transformer", transformer_config),
        ("vae", _VAE_CONFIG),
    ]:
        os.makedirs(os.path.join(config_dir, name))
        with open(os.path.join(config_dir, name, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    os.makedirs(os.path.join(config_dir, "scheduler"))
    with open(os.path.join(config_dir, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump(_SCHEDULER_CONFIG, f, indent=2)

    for name, repo in [
        ("text_encoder", "openai/clip-vit-large-patch14"),
        ("text_encoder_2", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"),
    ]:
        os.makedirs(os.path.join(config_dir, name))
        CLIPTextConfig.from_pretrained(repo).save_pretrained(
            os.path.join(config_dir, name)
        )

    os.makedirs(os.path.join(config_dir, "text_encoder_3"))
    AutoConfig.from_pretrained("google/t5-v1_1-xxl").save_pretrained(
        os.path.join(config_dir, "text_encoder_3")
    )

    CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14").save_pretrained(
        os.path.join(config_dir, "tokenizer")
    )
    CLIPTokenizer.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    ).save_pretrained(os.path.join(config_dir, "tokenizer_2"))
    T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl").save_pretrained(
        os.path.join(config_dir, "tokenizer_3")
    )

    return config_dir


def load_pipe(filename, dtype=torch.float32):
    """Load Stable Diffusion 3.5 FP8 pipeline from a single-file safetensors checkpoint.

    Args:
        filename: Safetensors filename within the Comfy-Org/stable-diffusion-3.5-fp8 repo.
        dtype: Torch dtype for the pipeline.

    Returns:
        StableDiffusion3Pipeline: Loaded pipeline with components set to eval mode.
    """
    is_large = "large" in filename.lower()
    transformer_config = (
        _LARGE_TRANSFORMER_CONFIG if is_large else _MEDIUM_TRANSFORMER_CONFIG
    )
    config_dir = _build_config_dir(transformer_config)

    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    pipe = StableDiffusion3Pipeline.from_single_file(
        checkpoint_path,
        config=config_dir,
        torch_dtype=dtype,
    )
    if pipe.text_encoder_3 is not None:
        t5 = pipe.text_encoder_3
        if t5.encoder.embed_tokens.weight.device.type == "meta":
            t5.encoder.embed_tokens.weight = t5.shared.weight

    modules = [
        pipe.text_encoder,
        pipe.transformer,
        pipe.text_encoder_2,
        pipe.vae,
    ]
    if pipe.text_encoder_3 is not None:
        modules.append(pipe.text_encoder_3)

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
    """Calculate dynamic shifting parameter for the scheduler."""
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    mu = image_seq_len * m + b
    return mu


def stable_diffusion_preprocessing_v35(
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
    """Preprocess inputs for Stable Diffusion 3.5 FP8 model.

    Args:
        pipe: Stable Diffusion 3.5 pipeline.
        prompt: Text prompt for generation.
        device: Device to run on.
        negative_prompt: Negative prompt (optional).
        guidance_scale: Guidance scale.
        num_inference_steps: Number of inference steps.
        num_images_per_prompt: Number of images per prompt.
        clip_skip: CLIP skip layers (optional).
        max_sequence_length: Maximum sequence length.
        do_classifier_free_guidance: Whether to use classifier-free guidance.
        mu: Dynamic shifting parameter (optional).

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds)
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
        image_seq_len = (height // pipe.transformer.config.patch_size) * (
            width // pipe.transformer.config.patch_size
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

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=1,
        device=device,
        sigmas=None,
        **scheduler_kwargs,
    )

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds
