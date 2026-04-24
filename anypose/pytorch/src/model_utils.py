# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for AnyPose model loading and processing.
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image


def load_anypose_pipe(base_model_name, lora_model_name, lora_scale=0.7):
    """Load AnyPose pipeline with LoRA weights.

    Args:
        base_model_name: Base Qwen Image Edit model name on HuggingFace
        lora_model_name: AnyPose LoRA model name on HuggingFace
        lora_scale: LoRA weight scale (default: 0.7)

    Returns:
        QwenImageEditPlusPipeline: Loaded pipeline with LoRA adapters
    """
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        base_model_name, torch_dtype=torch.float32
    )

    pipe.load_lora_weights(
        lora_model_name,
        weight_name="2511-AnyPose-base-000006250.safetensors",
        adapter_name="anypose_base",
    )
    pipe.load_lora_weights(
        lora_model_name,
        weight_name="2511-AnyPose-helper-00006000.safetensors",
        adapter_name="anypose_helper",
    )
    pipe.set_adapters(
        ["anypose_base", "anypose_helper"],
        adapter_weights=[lora_scale, lora_scale],
    )

    pipe.to("cpu")

    for component_name in ["text_encoder", "transformer", "vae"]:
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.eval()
            for param in component.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    return pipe


def create_dummy_images():
    """Create dummy input images for AnyPose inference.

    Returns:
        tuple: (character_image, pose_image) - Two PIL Images
    """
    character_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    pose_image = Image.new("RGB", (512, 512), color=(64, 64, 64))
    return character_image, pose_image


def anypose_preprocessing(pipe, prompt, character_image, pose_image):
    """Preprocess inputs for AnyPose model.

    Args:
        pipe: QwenImageEditPlusPipeline
        prompt: Text prompt for pose transfer
        character_image: PIL Image of the character to modify
        pose_image: PIL Image with the reference pose

    Returns:
        dict: Preprocessed inputs for the pipeline
    """
    inputs = {
        "image": [character_image, pose_image],
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    return inputs


def create_transformer_inputs(
    transformer,
    image_size=(512, 512),
    batch_size=1,
    num_input_images=2,
    text_seq_len=64,
):
    """Create synthetic inputs for QwenImageTransformer2DModel.forward().

    For a 512x512 image with VAE scale factor 8 and patch size 2:
    - Latent: 64x64 → packed 2x2 patches → 1024 tokens per image
    - hidden_states shape: (batch, tokens_noise + tokens_img1 + tokens_img2, in_channels)

    Args:
        transformer: QwenImageTransformer2DModel instance
        image_size: Input image size as (height, width)
        batch_size: Batch size
        num_input_images: Number of conditioning images (AnyPose uses 2)
        text_seq_len: Text sequence length for synthetic encoder hidden states

    Returns:
        dict: Inputs matching QwenImageTransformer2DModel.forward() signature
    """
    height, width = image_size
    vae_scale_factor = 8
    patch_size = transformer.config.patch_size
    in_channels = transformer.config.in_channels
    joint_attention_dim = transformer.config.joint_attention_dim

    lat_h = height // vae_scale_factor
    lat_w = width // vae_scale_factor
    tokens_per_image = (lat_h // patch_size) * (lat_w // patch_size)
    total_tokens = tokens_per_image * (1 + num_input_images)

    lat_h_packed = lat_h // patch_size
    lat_w_packed = lat_w // patch_size
    img_shapes = [
        [(1, lat_h_packed, lat_w_packed)] * (1 + num_input_images)
    ] * batch_size

    hidden_states = torch.randn(
        batch_size, total_tokens, in_channels, dtype=torch.float32
    )
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, joint_attention_dim, dtype=torch.float32
    )
    encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool)
    timestep = torch.tensor([0.5] * batch_size, dtype=torch.float32)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": encoder_hidden_states_mask,
        "timestep": timestep,
        "img_shapes": img_shapes,
        "return_dict": False,
    }
