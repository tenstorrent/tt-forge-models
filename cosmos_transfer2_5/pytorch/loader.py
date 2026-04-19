# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Transfer2.5-2B model loader for tt_forge_models.

Cosmos Transfer2.5 is a 2.36B-parameter diffusion transformer by NVIDIA for
controlled video generation (video-to-video). It takes control signals (Canny
edge, depth maps, segmentation masks, or blurred RGB) plus a text prompt and
generates 1280x720 video at 16 FPS.

Repository:
- https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B

The HuggingFace repo is gated, so models are constructed from config with
random weights for compile-only testing.
"""

from typing import Any, Optional

import torch
from diffusers import CosmosControlNetModel, CosmosTransformer3DModel

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

COSMOS_TRANSFER_REPO = "nvidia/Cosmos-Transfer2.5-2B"

_TRANSFORMER_CONFIG = {
    "in_channels": 16,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": (128, 240, 240),
    "patch_size": (1, 2, 2),
    "rope_scale": (2.0, 1.0, 1.0),
    "concat_padding_mask": False,
    "extra_pos_embed_type": "learnable",
}

_CONTROLNET_CONFIG = {
    "n_controlnet_blocks": 4,
    "in_channels": 130,
    "latent_channels": 18,
    "model_channels": 2048,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "patch_size": (1, 2, 2),
    "max_size": (128, 240, 240),
    "rope_scale": (2.0, 1.0, 1.0),
}


class ModelVariant(StrEnum):
    """Available Cosmos Transfer2.5 variants."""

    EDGE = "edge"
    DEPTH = "depth"
    SEG = "seg"
    BLUR = "blur"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.EDGE: ModelConfig(
            pretrained_model_name=COSMOS_TRANSFER_REPO,
        ),
        ModelVariant.DEPTH: ModelConfig(
            pretrained_model_name=COSMOS_TRANSFER_REPO,
        ),
        ModelVariant.SEG: ModelConfig(
            pretrained_model_name=COSMOS_TRANSFER_REPO,
        ),
        ModelVariant.BLUR: ModelConfig(
            pretrained_model_name=COSMOS_TRANSFER_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EDGE

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = None,
    ):
        super().__init__(variant)
        self._subfolder = subfolder
        self._transformer = None
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Cosmos-Transfer2.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_controlnet(self, dtype: torch.dtype):
        self._controlnet = CosmosControlNetModel(**_CONTROLNET_CONFIG).to(dtype)
        return self._controlnet

    def _load_transformer(self, dtype: torch.dtype):
        self._transformer = CosmosTransformer3DModel(**_TRANSFORMER_CONFIG).to(dtype)
        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "controlnet":
            if self._controlnet is None:
                self._load_controlnet(dtype)
            return self._controlnet

        if self._transformer is None:
            self._load_transformer(dtype)
        return self._transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "controlnet":
            return self._load_controlnet_inputs(dtype)
        return self._load_transformer_inputs(dtype)

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        if self._transformer is None:
            self._load_transformer(dtype)

        batch_size = 1
        config = self._transformer.config

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            2,
            2,
            2,
            dtype=dtype,
        )

        encoder_hidden_states = torch.randn(
            batch_size, 8, config.text_embed_dim, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def _load_controlnet_inputs(self, dtype: torch.dtype) -> dict:
        if self._controlnet is None:
            self._load_controlnet(dtype)

        config = self._controlnet.config
        batch_size = 1
        latent_channels = config.latent_channels

        latents = torch.randn(
            batch_size,
            latent_channels,
            2,
            2,
            2,
            dtype=dtype,
        )

        controls_latents = torch.randn(
            batch_size,
            latent_channels,
            2,
            2,
            2,
            dtype=dtype,
        )

        encoder_hidden_states = torch.randn(
            batch_size, 8, config.text_embed_dim, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        condition_mask = torch.ones(batch_size, 1, dtype=dtype)

        return {
            "latents": latents,
            "controls_latents": controls_latents,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "condition_mask": condition_mask,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
