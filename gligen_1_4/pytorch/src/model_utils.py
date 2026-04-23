# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for GLIGEN 1.4 model loading and preprocessing."""

import torch


def gligen_preprocessing(
    pipe,
    prompt,
    gligen_phrases,
    gligen_boxes,
    device="cpu",
    num_images_per_prompt=1,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=None,
    width=None,
):
    """Preprocess inputs for a single GLIGEN UNet forward pass.

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds, cross_attention_kwargs)
    """
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    do_classifier_free_guidance = guidance_scale > 1.0

    if isinstance(prompt, str):
        batch_size = 1
    else:
        batch_size = len(prompt)

    # Encode text prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Prepare latents
    num_channels_latents = pipe.unet.config.in_channels
    torch.manual_seed(42)
    latents = torch.randn(
        (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
        dtype=prompt_embeds.dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    # Prepare GLIGEN grounding variables
    max_objs = 30
    phrases_flat = (
        gligen_phrases[0] if isinstance(gligen_phrases[0], list) else gligen_phrases
    )
    boxes_flat = gligen_boxes[0] if isinstance(gligen_boxes[0], list) else gligen_boxes

    tokenizer_inputs = pipe.tokenizer(
        phrases_flat, padding=True, return_tensors="pt"
    ).to(device)
    _text_embeddings = pipe.text_encoder(**tokenizer_inputs).pooler_output

    n_objs = len(boxes_flat)
    boxes_tensor = torch.zeros(
        max_objs, 4, device=device, dtype=pipe.text_encoder.dtype
    )
    boxes_tensor[:n_objs] = torch.tensor(
        boxes_flat, device=device, dtype=pipe.text_encoder.dtype
    )
    text_embeddings = torch.zeros(
        max_objs,
        pipe.unet.config.cross_attention_dim,
        device=device,
        dtype=pipe.text_encoder.dtype,
    )
    text_embeddings[:n_objs] = _text_embeddings
    masks = torch.zeros(max_objs, device=device, dtype=pipe.text_encoder.dtype)
    masks[:n_objs] = 1

    repeat_batch = batch_size * num_images_per_prompt
    boxes_tensor = boxes_tensor.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
    text_embeddings = text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
    masks = masks.unsqueeze(0).expand(repeat_batch, -1).clone()

    if do_classifier_free_guidance:
        repeat_batch = repeat_batch * 2
        boxes_tensor = torch.cat([boxes_tensor] * 2)
        text_embeddings = torch.cat([text_embeddings] * 2)
        masks = torch.cat([masks] * 2)
        masks[: repeat_batch // 2] = 0

    cross_attention_kwargs = {
        "gligen": {
            "boxes": boxes_tensor,
            "positive_embeddings": text_embeddings,
            "masks": masks,
        }
    }

    # Build a single-step latent input
    t = timesteps[0]
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    return latent_model_input, t, prompt_embeds, cross_attention_kwargs
