# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import QwenImageEditPlusPipeline


def load_pipeline(pretrained_model_name, dtype=torch.float32):
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        pretrained_model_name,
        torch_dtype=dtype,
    )
    pipe.to("cpu")
    pipe.transformer.eval()
    for param in pipe.transformer.parameters():
        param.requires_grad = False
    return pipe


def preprocess_inputs(pipeline, prompt, image, dtype=torch.float32):
    device = "cpu"
    batch_size = 1
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    vae_scale_factor = pipeline.vae_scale_factor

    image_size = image.size
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        calculate_dimensions,
        CONDITION_IMAGE_SIZE,
        VAE_IMAGE_SIZE,
    )

    target_w, target_h = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    multiple_of = vae_scale_factor * 2
    target_w = target_w // multiple_of * multiple_of
    target_h = target_h // multiple_of * multiple_of

    condition_w, condition_h = calculate_dimensions(
        CONDITION_IMAGE_SIZE, image_size[0] / image_size[1]
    )
    vae_w, vae_h = calculate_dimensions(
        VAE_IMAGE_SIZE, image_size[0] / image_size[1]
    )

    condition_image = pipeline.image_processor.resize(image, condition_h, condition_w)

    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
        image=[condition_image],
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
    )

    target_lat_h = 2 * (int(target_h) // (vae_scale_factor * 2))
    target_lat_w = 2 * (int(target_w) // (vae_scale_factor * 2))

    target_shape = (batch_size, 1, num_channels_latents, target_lat_h, target_lat_w)
    target_latents = torch.randn(target_shape, dtype=dtype, device=device)
    target_latents = pipeline._pack_latents(
        target_latents, batch_size, num_channels_latents, target_lat_h, target_lat_w
    )

    src_lat_h = 2 * (int(vae_h) // (vae_scale_factor * 2))
    src_lat_w = 2 * (int(vae_w) // (vae_scale_factor * 2))
    src_shape = (batch_size, 1, num_channels_latents, src_lat_h, src_lat_w)
    image_latents = torch.randn(src_shape, dtype=dtype, device=device)
    image_latents = pipeline._pack_latents(
        image_latents, batch_size, num_channels_latents, src_lat_h, src_lat_w
    )

    hidden_states = torch.cat([target_latents, image_latents], dim=1)

    timestep = torch.tensor([1.0], dtype=dtype, device=device)

    img_shapes = [
        [
            (1, target_lat_h // 2, target_lat_w // 2),
            (1, src_lat_h // 2, src_lat_w // 2),
        ]
    ]

    inputs = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": prompt_embeds,
        "encoder_hidden_states_mask": prompt_embeds_mask,
        "img_shapes": img_shapes,
    }

    return inputs
