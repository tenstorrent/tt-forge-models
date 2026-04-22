# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Fill-dev GGUF model loader implementation for image inpainting.

This loader uses GGUF-quantized variants of the FLUX.1-Fill-dev transformer from
YarvixPA/FLUX.1-Fill-dev-GGUF. The transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file with a locally constructed config to avoid
requiring access to the gated black-forest-labs/FLUX.1-Fill-dev repository.
"""

import json
import os
import shutil
import tempfile
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

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

GGUF_REPO = "YarvixPA/FLUX.1-Fill-dev-GGUF"

# FLUX.1-Fill-dev transformer config (in_channels=384: 64 latent + 64 masked + 256 mask)
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "attention_head_dim": 128,
    "guidance_embeds": True,
    "in_channels": 384,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "out_channels": 16,
    "patch_size": 2,
    "pooled_projection_dim": 768,
}

_VAE_SCALE_FACTOR = 16


class ModelVariant(StrEnum):
    """Available FLUX.1-Fill-dev GGUF quantization variants."""

    Q3_K_S = "Q3_K_S"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K_S = "Q4_K_S"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K_S = "Q5_K_S"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q3_K_S: "flux1-fill-dev-Q3_K_S.gguf",
    ModelVariant.Q4_0: "flux1-fill-dev-Q4_0.gguf",
    ModelVariant.Q4_1: "flux1-fill-dev-Q4_1.gguf",
    ModelVariant.Q4_K_S: "flux1-fill-dev-Q4_K_S.gguf",
    ModelVariant.Q5_0: "flux1-fill-dev-Q5_0.gguf",
    ModelVariant.Q5_1: "flux1-fill-dev-Q5_1.gguf",
    ModelVariant.Q5_K_S: "flux1-fill-dev-Q5_K_S.gguf",
    ModelVariant.Q6_K: "flux1-fill-dev-Q6_K.gguf",
    ModelVariant.Q8_0: "flux1-fill-dev-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX.1-Fill-dev GGUF model loader for image inpainting."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-Fill-dev GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        """Load the FLUX Fill transformer from a GGUF file."""
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)

        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        # Write a local config so from_single_file doesn't need the gated BFL repo
        config_dir = tempfile.mkdtemp()
        try:
            config_path = os.path.join(config_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_TRANSFORMER_CONFIG, f)

            self.transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                config=config_dir,
                quantization_config=quantization_config,
                torch_dtype=dtype,
            )
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX Fill transformer.

        Returns:
            torch.nn.Module: The FLUX Fill transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.transformer is None:
            self._load_transformer(dtype)
        elif dtype_override is not None:
            self.transformer = self.transformer.to(dtype=dtype_override)
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX Fill transformer.

        The FLUX Fill transformer expects hidden_states that include the noisy latents
        concatenated with masked image latents and a packed mask along dim=2,
        resulting in in_channels of 384 (64 + 64 + 256).

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.transformer is None:
            self._load_transformer(dtype)

        config = self.transformer.config

        height = 128
        width = 128
        height_latent = 2 * (height // (_VAE_SCALE_FACTOR * 2))
        width_latent = 2 * (width // (_VAE_SCALE_FACTOR * 2))
        seq_len = (height_latent // 2) * (width_latent // 2)

        # FLUX.1-Fill-dev hidden_states: noisy latents + masked image + packed mask
        hidden_states = torch.randn(
            batch_size, seq_len, config.in_channels, dtype=dtype
        )

        # Latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        max_sequence_length = 256

        # Synthetic text embeddings (avoids loading gated text encoder models)
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, config.joint_attention_dim, dtype=dtype
        )
        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # FLUX.1-Fill-dev uses classifier-free guidance
        guidance = torch.tensor([3.5], dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
