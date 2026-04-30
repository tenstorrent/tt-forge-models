# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-schnell GGUF (leejet/FLUX.1-schnell-gguf) model loader implementation.

FLUX.1-schnell is a 12B parameter text-to-image generation model in GGUF quantized format,
based on the FLUX transformer architecture from Black Forest Labs.

The transformer is loaded via diffusers' FluxTransformer2DModel.from_single_file with a
locally materialised config so that the loader does not depend on the gated
black-forest-labs/FLUX.1-schnell repository. GGUF weights are immediately dequantized to
plain bf16 nn.Linear so that TorchDynamo tracing does not recurse into
GGUFParameter.__torch_function__.

Available variants:
- FLUX1_SCHNELL_Q4_0: Q4_0 quantized variant (~6.88 GB)
"""

import json
import os
import tempfile
from typing import Optional

import torch
from diffusers import GGUFQuantizationConfig
from diffusers.models import FluxTransformer2DModel
from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

GGUF_REPO = "leejet/FLUX.1-schnell-gguf"

# FLUX.1-schnell transformer architecture (no guidance embeddings, unlike dev).
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available FLUX.1-schnell GGUF model variants."""

    FLUX1_SCHNELL_Q4_0 = "flux1_schnell_Q4_0"


_GGUF_FILES = {
    ModelVariant.FLUX1_SCHNELL_Q4_0: "flux1-schnell-q4_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX.1-schnell GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX1_SCHNELL_Q4_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX1_SCHNELL_Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-schnell GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self):
        """Write a minimal transformer/config.json to a temp dir for from_single_file."""
        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(_TRANSFORMER_CONFIG, f)
        return config_dir

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        """Load and dequantize the FluxTransformer2DModel from the GGUF file."""
        gguf_file = _GGUF_FILES[self._variant]
        gguf_url = f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}"
        config_dir = self._make_local_config_dir()

        self._transformer = FluxTransformer2DModel.from_single_file(
            gguf_url,
            config=config_dir,
            subfolder="transformer",
            torch_dtype=dtype,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        )
        # GGUFParameter.__torch_function__ recurses under TorchDynamo tracing;
        # dequantize to plain tensors before compilation.
        _dequantize_gguf_and_restore_linear(self._transformer)
        self._transformer.is_quantized = False
        self._transformer = self._transformer.to(dtype=dtype)
        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX.1-schnell transformer from GGUF checkpoint.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            self._load_transformer(dtype)
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare synthetic inputs for the FLUX transformer.

        Returns:
            dict: Input tensors for the FLUX transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            self._load_transformer(dtype)

        config = self._transformer.config
        max_sequence_length = 256
        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        # Pack latents to (B, H*W, C).
        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Latent image IDs.
        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Synthetic text embeddings (no tokenizer or text encoder needed).
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, config.joint_attention_dim, dtype=dtype
        )
        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": None,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
