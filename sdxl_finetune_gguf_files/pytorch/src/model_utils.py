# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Stable Diffusion XL finetune models.
"""

from typing import Optional

import torch
from diffusers import UNet2DConditionModel
from huggingface_hub import hf_hub_download

# SDXL base model repo used only for UNet config (no weight download).
_SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"


def _load_comfyui_gguf_state_dict(gguf_path: str) -> dict:
    """Load a ComfyUI-format GGUF file into a plain FP32 state dict.

    ComfyUI GGUF files:
    - Store tensor data with GGUFReader-transposed shapes (``t.data.shape``).
    - May include ``comfy.gguf.orig_shape.*`` metadata when the transposed shape
      differs from the original PyTorch shape (e.g. non-square weight matrices
      that were quantized into a block-aligned layout).
    - Quantized tensors must be dequantized; the dequantized flat tensor is then
      reshaped to the stored orig_shape.
    - F16/F32 tensors either already have the correct shape after transposition
      by GGUFReader, or carry an orig_shape for the same reason as quantized ones.
    """
    import gguf as gguf_module
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)

    # Collect per-tensor original shapes from ComfyUI metadata.
    orig_shapes: dict[str, tuple] = {}
    for key, field in reader.fields.items():
        if key.startswith("comfy.gguf.orig_shape."):
            tensor_name = key[len("comfy.gguf.orig_shape.") :]
            orig_shapes[tensor_name] = tuple(
                int(field.parts[idx][0]) for idx in field.data
            )

    state_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type

        is_gguf_quant = quant_type not in (
            gguf_module.GGMLQuantizationType.F32,
            gguf_module.GGMLQuantizationType.F16,
        )

        data = torch.from_numpy(tensor.data.copy())

        if is_gguf_quant:
            param = GGUFParameter(data, quant_type=quant_type)
            data = dequantize_gguf_tensor(param).to(torch.float32)
        else:
            data = data.to(torch.float32)

        # Reshape to the original model shape if ComfyUI stored a hint.
        if name in orig_shapes:
            data = data.reshape(orig_shapes[name])

        state_dict[name] = data

    return state_dict


def load_gguf_unet(repo_id: str, gguf_filename: str, subfolder: Optional[str] = None):
    """Load an SDXL UNet from a ComfyUI-format GGUF checkpoint.

    The file contains only the UNet weights (no text encoders/VAE) with keys
    in the original LDM naming convention (without the ``model.diffusion_model.``
    prefix expected by ``convert_ldm_unet_checkpoint``).

    Args:
        repo_id: HuggingFace repository ID.
        gguf_filename: Filename of the GGUF checkpoint within the repo.
        subfolder: Optional subfolder within the repo.

    Returns:
        UNet2DConditionModel: Loaded UNet in eval mode with frozen weights.
    """
    model_path = hf_hub_download(
        repo_id=repo_id, filename=gguf_filename, subfolder=subfolder
    )

    state_dict = _load_comfyui_gguf_state_dict(model_path)

    # convert_ldm_unet_checkpoint expects the "model.diffusion_model." prefix.
    prefixed = {"model.diffusion_model." + k: v for k, v in state_dict.items()}

    unet = UNet2DConditionModel.from_single_file(
        prefixed,
        config=_SDXL_BASE_REPO,
        subfolder="unet",
        torch_dtype=torch.float32,
    )
    unet.eval()
    for param in unet.parameters():
        param.requires_grad = False

    return unet


def make_sdxl_unet_inputs(unet, dtype=torch.float32):
    """Generate synthetic UNet inputs from the UNet config.

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds, added_cond_kwargs)
    """
    cfg = unet.config
    batch = 2  # classifier-free guidance doubles the batch
    latent_h = 128  # 1024 // 8
    latent_w = 128

    latent_model_input = torch.randn(
        batch, cfg.in_channels, latent_h, latent_w, dtype=dtype
    )
    timestep = torch.tensor([999], dtype=torch.long)
    prompt_embeds = torch.randn(batch, 77, cfg.cross_attention_dim, dtype=dtype)

    # SDXL time-embedding: text_embeds (1280) + 6 time_ids sinusoidal-embedded.
    text_embeds = torch.randn(batch, 1280, dtype=dtype)
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * batch, dtype=dtype)
    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

    return latent_model_input, timestep, prompt_embeds, added_cond_kwargs
