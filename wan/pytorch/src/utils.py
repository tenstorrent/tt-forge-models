# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Wan model loading (VAE and VACE pipeline)."""

import torch
from PIL import Image


# Wan VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Small test dimensions for VAE inputs
# Wan VAE compression: 4x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames

# Small test dimensions for VACE transformer inputs
# Height and width must be divisible by patch_size (2, 2)
VACE_TRANSFORMER_NUM_FRAMES = 1
VACE_TRANSFORMER_HEIGHT = 8
VACE_TRANSFORMER_WIDTH = 8
VACE_TRANSFORMER_TEXT_SEQ_LEN = 16


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load AutoencoderKLWan from diffusers.

    Args:
        pretrained_model_name: HuggingFace model ID (e.g. "Wan-AI/Wan2.1-T2V-14B-Diffusers")
        dtype: Torch dtype for model weights
    """
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.eval()

    return vae


# ============================================================================
# Input Loading Functions
# ============================================================================


def load_vae_decoder_inputs(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load inputs for VAE decoder.

    Args:
        dtype: Data type for the tensor

    Returns:
        Latent tensor of shape [1, 16, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH]
    """
    # [batch, channels, time, height, width]
    return torch.randn(
        1, LATENT_CHANNELS, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
    )


def load_vae_encoder_inputs(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load inputs for VAE encoder.

    Wan VAE requires frame count T = 1 + 4*N for some integer N.

    Args:
        dtype: Data type for the tensor

    Returns:
        RGB video tensor of shape [1, 3, T, H, W]
        where T = 1 + 4*LATENT_DEPTH, H = LATENT_HEIGHT*8, W = LATENT_WIDTH*8
    """
    # T must satisfy T = 1 + 4*N (Wan temporal constraint)
    num_frames = 1 + 4 * LATENT_DEPTH  # 9 frames
    return torch.randn(
        1, 3, num_frames, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype
    )


# ============================================================================
# VACE Pipeline Loading Functions
# ============================================================================


def load_vace_pipeline(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanVACEPipeline from diffusers.

    Args:
        pretrained_model_name: HuggingFace model ID
            (e.g. "Wan-AI/Wan2.1-VACE-1.3B-diffusers")
        dtype: Torch dtype for model weights
    """
    from diffusers import AutoencoderKLWan, WanVACEPipeline

    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanVACEPipeline.from_pretrained(
        pretrained_model_name,
        vae=vae,
        torch_dtype=dtype,
    )
    return pipe


# ============================================================================
# I2V Pipeline Loading Functions
# ============================================================================


def load_i2v_pipeline(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanImageToVideoPipeline from diffusers.

    The image encoder and VAE are loaded in float32 for numerical stability,
    while the main transformer uses the provided dtype.

    Args:
        pretrained_model_name: HuggingFace model ID
            (e.g. "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
        dtype: Torch dtype for the transformer weights
    """
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    image_encoder = CLIPVisionModel.from_pretrained(
        pretrained_model_name,
        subfolder="image_encoder",
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


def load_i2v_inputs(prompt: str, height: int = 480, width: int = 832) -> dict:
    """
    Prepare inputs for the I2V pipeline (image-to-video generation).

    Returns a dict suitable for passing to WanImageToVideoPipeline.__call__.
    Uses a small synthetic image for testing.

    Args:
        prompt: Text prompt for generation
        height: Output video height (default 480 for 480P, use 720 for 720P)
        width: Output video width (default 832 for 480P, use 1280 for 720P)
    """
    ref_image = Image.new("RGB", (width, height), color=(128, 128, 200))

    return {
        "image": ref_image,
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": 9,
        "num_inference_steps": 2,
        "guidance_scale": 5.0,
    }


def load_vace_transformer_inputs(
    transformer, dtype: torch.dtype = torch.float32
) -> dict:
    """
    Prepare tensor inputs for the WanVACETransformer3DModel forward pass.

    Args:
        transformer: Loaded WanVACETransformer3DModel instance
        dtype: Torch dtype for generated tensors
    """
    config = transformer.config
    batch_size = 1

    hidden_states = torch.randn(
        batch_size,
        config.in_channels,
        VACE_TRANSFORMER_NUM_FRAMES,
        VACE_TRANSFORMER_HEIGHT,
        VACE_TRANSFORMER_WIDTH,
        dtype=dtype,
    )
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_hidden_states = torch.randn(
        batch_size,
        VACE_TRANSFORMER_TEXT_SEQ_LEN,
        config.text_dim,
        dtype=dtype,
    )
    control_hidden_states = torch.randn(
        batch_size,
        config.vace_in_channels,
        VACE_TRANSFORMER_NUM_FRAMES,
        VACE_TRANSFORMER_HEIGHT,
        VACE_TRANSFORMER_WIDTH,
        dtype=dtype,
    )
    control_hidden_states_scale = torch.ones(len(config.vace_layers), dtype=dtype)

    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "control_hidden_states": control_hidden_states,
        "control_hidden_states_scale": control_hidden_states_scale,
        "return_dict": False,
    }


def load_vace_inputs(prompt: str) -> dict:
    """
    Prepare inputs for the VACE pipeline (reference-to-video generation).

    Returns a dict suitable for passing to WanVACEPipeline.__call__.
    Uses a small synthetic reference image for testing.
    """
    # Create a small synthetic reference image for R2V (reference-to-video)
    ref_image = Image.new("RGB", (832, 480), color=(128, 128, 200))

    return {
        "prompt": prompt,
        "reference_images": [ref_image],
        "height": 480,
        "width": 832,
        "num_frames": 9,
        "num_inference_steps": 2,
        "guidance_scale": 5.0,
    }
