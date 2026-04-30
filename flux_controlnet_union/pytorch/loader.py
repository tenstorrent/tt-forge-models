# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX ControlNet Union model loader implementation.

Tests the FLUX.1-dev transformer architecture on TT silicon. Weights are loaded
from a GGUF-quantized checkpoint (InvokeAI/FLUX.1-Krea-dev-GGUF Q4_K_S) to
avoid the gated black-forest-labs/FLUX.1-dev repository. GGUFParameter tensor
subclasses are eagerly dequantized after loading so the model has plain
floating-point parameters compatible with TT-silicon compilation.
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
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

_GGUF_REPO = "InvokeAI/FLUX.1-Krea-dev-GGUF"
_GGUF_FILE = "flux1-krea-dev-Q4_K_S.gguf"

# Standard FLUX.1-dev transformer architecture parameters.
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available FLUX ControlNet Union model variants."""

    FLUX_1_DEV_CONTROLNET_UNION = "FLUX.1-dev-Controlnet-Union"


class ModelLoader(ForgeModel):
    """FLUX ControlNet Union model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX_1_DEV_CONTROLNET_UNION: ModelConfig(
            pretrained_model_name="InstantX/FLUX.1-dev-Controlnet-Union",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX_1_DEV_CONTROLNET_UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX ControlNet Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self):
        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(_TRANSFORMER_CONFIG, f)
        return config_dir

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        gguf_url = f"https://huggingface.co/{_GGUF_REPO}/blob/main/{_GGUF_FILE}"
        config_dir = self._make_local_config_dir()

        self._transformer = FluxTransformer2DModel.from_single_file(
            gguf_url,
            config=config_dir,
            subfolder="transformer",
            torch_dtype=dtype,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        )
        _dequantize_gguf_and_restore_linear(self._transformer)
        self._transformer.is_quantized = False
        self._transformer = self._transformer.to(dtype=dtype)
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX transformer model.

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
        """Load and return sample inputs for the FLUX transformer.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
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

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, config.joint_attention_dim, dtype=dtype
        )
        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)
        guidance = torch.full([batch_size], 3.5, dtype=dtype)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
