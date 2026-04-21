# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for WAMU I2V Lightning model loading."""

import torch


def load_i2v_pipeline(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanImageToVideoPipeline from diffusers.

    The image encoder and VAE are loaded in float32 for numerical stability,
    while the main transformer uses the provided dtype.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for the transformer weights
    """
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionConfig, CLIPVisionModel

    clip_config = CLIPVisionConfig.from_pretrained(
        pretrained_model_name,
        subfolder="image_encoder",
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        pretrained_model_name,
        subfolder="image_encoder",
        config=clip_config,
        torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        pretrained_model_name,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    return pipe


def wan_i2v_preprocessing(pipeline, prompt, device="cpu"):
    """Preprocess inputs for the WanTransformer3DModel forward pass.

    Args:
        pipeline: WanImageToVideoPipeline instance
        prompt: Text prompt for generation
        device: Device to run on

    Returns:
        dict: Keyword arguments for the transformer forward method
    """
    height, width, num_frames = 480, 832, 9
    batch_size = 1
    transformer = pipeline.transformer
    transformer_dtype = transformer.dtype

    prompt_embeds, _ = pipeline.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        device=device,
    )
    prompt_embeds = prompt_embeds.to(transformer_dtype)

    in_channels = transformer.config.in_channels
    vae_scale_spatial = pipeline.vae_scale_factor_spatial
    vae_scale_temporal = pipeline.vae_scale_factor_temporal
    latent_frames = (num_frames - 1) // vae_scale_temporal + 1
    latent_h = height // vae_scale_spatial
    latent_w = width // vae_scale_spatial

    latent_model_input = torch.randn(
        batch_size,
        in_channels,
        latent_frames,
        latent_h,
        latent_w,
        device=device,
        dtype=transformer_dtype,
    )

    expand_timesteps = getattr(pipeline.config, "expand_timesteps", False)
    if expand_timesteps:
        p_t, p_h, p_w = transformer.config.patch_size
        seq_len = latent_frames * (latent_h // p_h) * (latent_w // p_w)
        timestep = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    else:
        timestep = torch.tensor([500], dtype=torch.long, device=device).expand(
            batch_size
        )

    result = {
        "hidden_states": latent_model_input,
        "timestep": timestep,
        "encoder_hidden_states": prompt_embeds,
    }

    if transformer.config.image_dim is not None:
        image_encoder = pipeline.image_encoder
        clip_image_size = image_encoder.config.image_size
        pixel_values = torch.randn(
            batch_size,
            3,
            clip_image_size,
            clip_image_size,
            device=device,
            dtype=image_encoder.dtype,
        )
        image_out = image_encoder(pixel_values=pixel_values, output_hidden_states=True)
        image_embeds = image_out.hidden_states[-2].to(transformer_dtype)
        result["encoder_hidden_states_image"] = image_embeds

    return result
