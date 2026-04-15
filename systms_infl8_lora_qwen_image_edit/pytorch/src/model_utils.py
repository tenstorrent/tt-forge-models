# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for SYSTMS INFL8 LoRA Qwen Image Edit model loading and preprocessing.
"""

import torch
from diffusers import DiffusionPipeline

# Transformer config constants from Qwen/Qwen-Image-Edit-2511
# in_channels=64, joint_attention_dim=3584, patch_size=2
# VAE: z_dim=16, dim_mult=[1,2,4,4] -> vae_scale_factor=8
_TRANSFORMER_IN_CHANNELS = 64  # packed latent channels
_TEXT_ENCODER_HIDDEN_SIZE = 3584  # joint_attention_dim
_VAE_SCALE_FACTOR = 8
_TARGET_IMAGE_SIZE = 1024  # default target for 1:1 aspect ratio images
_NUM_LATENT_CHANNELS = 16  # in_channels // 4


def load_pipe(base_model_name, lora_repo, lora_weights, dtype=torch.bfloat16):
    """Load the QwenImageEditPlus pipeline with LoRA weights applied.

    Args:
        base_model_name: HuggingFace model name for the base pipeline.
        lora_repo: HuggingFace repo for the LoRA weights.
        lora_weights: Filename of the LoRA weights.
        dtype: Torch dtype for model loading.

    Returns:
        DiffusionPipeline with LoRA weights loaded.
    """
    pipe = DiffusionPipeline.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
    )
    pipe.load_lora_weights(lora_repo, weight_name=lora_weights)
    pipe.to("cpu")

    for module in [pipe.transformer, pipe.text_encoder, pipe.vae]:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    return pipe


def qwen_image_edit_fake_inputs(pipe, dtype=torch.bfloat16):
    """Create random inputs for the QwenImageTransformer2DModel forward pass.

    Uses synthetic random tensors of the correct shapes to avoid the expensive
    CPU inference of the 7B text encoder during input preparation. This is
    appropriate for compile-only testing where only tensor shapes matter.

    The input shapes are derived from the transformer and VAE configs:
    - Target image: 1024x1024 (standard for 1:1 aspect ratio)
    - VAE scale factor: 8, z_dim: 16
    - Packed latent size: (1, 4096, 64) per image stream
    - Combined (noise + condition): (1, 8192, 64)
    - Text encoder hidden size: 3584 (Qwen2.5-VL)

    Args:
        pipe: Loaded QwenImageEditPlusPipeline (used for transformer config).
        dtype: Torch dtype for input tensors.

    Returns:
        dict: Keyword arguments for transformer.forward().
    """
    # With 1024x1024 target:
    # latent height/width after VAE (with prepare_latents logic):
    # height = 2 * (1024 // (vae_scale_factor * 2)) = 2 * 64 = 128
    # After packing (2x2 blocks): seq_len = (128//2) * (128//2) = 4096 patches
    latent_seq_len = (_TARGET_IMAGE_SIZE // _VAE_SCALE_FACTOR // 2) ** 2  # 4096

    # Combined noise + condition image latents
    combined_seq_len = latent_seq_len * 2  # 8192

    # Spatial dims of each image stream for RoPE
    spatial_size = _TARGET_IMAGE_SIZE // _VAE_SCALE_FACTOR // 2  # 64

    # img_shapes: list of lists of (temporal, height, width) tuples
    # One tuple per image stream (noise latent + condition image latent)
    img_shapes = [
        [
            (1, spatial_size, spatial_size),  # noise latent
            (1, spatial_size, spatial_size),  # condition image latent
        ]
    ]

    # Random latent hidden states: (batch, combined_seq_len, in_channels)
    hidden_states = torch.randn(
        1, combined_seq_len, _TRANSFORMER_IN_CHANNELS, dtype=dtype
    )

    # Random text encoder embeddings: (batch, text_seq_len, hidden_size)
    # Use 128 as representative sequence length
    text_seq_len = 128
    encoder_hidden_states = torch.randn(
        1, text_seq_len, _TEXT_ENCODER_HIDDEN_SIZE, dtype=dtype
    )

    # Normalized timestep in [0, 1]
    timestep = torch.tensor([0.9], dtype=dtype)

    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": None,
        "img_shapes": img_shapes,
        "guidance": None,
        "return_dict": False,
    }
