# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image InstantX ControlNets model loader implementation.

Loads FluxControlNetModel variants in diffusers format.
Supports Inpainting and Union ControlNet variants.

Available variants:
- INPAINTING: ControlNet for image inpainting
- UNION: ControlNet supporting multiple control modes
"""

from typing import Any, Optional

import torch
from diffusers import FluxControlNetModel

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

_VARIANT_REPOS = {
    "inpainting": "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    "union": "InstantX/FLUX.1-dev-Controlnet-Union",
}


class ModelVariant(StrEnum):
    INPAINTING = "Inpainting"
    UNION = "Union"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.INPAINTING: ModelConfig(
            pretrained_model_name=_VARIANT_REPOS["inpainting"],
        ),
        ModelVariant.UNION: ModelConfig(
            pretrained_model_name=_VARIANT_REPOS["union"],
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_INSTANTX_CONTROLNETS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_version_key(self) -> str:
        return {
            ModelVariant.INPAINTING: "inpainting",
            ModelVariant.UNION: "union",
        }[self._variant]

    def _load_controlnet(
        self, dtype: torch.dtype = torch.float32
    ) -> FluxControlNetModel:
        version = self._get_version_key()
        repo_id = _VARIANT_REPOS[version]

        self._controlnet = FluxControlNetModel.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
        self._controlnet.eval()
        return self._controlnet

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._controlnet is None:
            return self._load_controlnet(dtype)
        if dtype_override is not None:
            self._controlnet = self._controlnet.to(dtype=dtype_override)
        return self._controlnet

    def load_inputs(self, **kwargs) -> Any:
        dtype = kwargs.get("dtype_override", self._controlnet.dtype)
        batch_size = kwargs.get("batch_size", 1)

        in_channels = self._controlnet.config["in_channels"]
        joint_attention_dim = self._controlnet.config["joint_attention_dim"]
        pooled_projection_dim = self._controlnet.config["pooled_projection_dim"]

        img_seq_len = 64
        txt_seq_len = 32

        hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        controlnet_cond = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_ids = torch.zeros(img_seq_len, 3, dtype=dtype)
        txt_ids = torch.zeros(txt_seq_len, 3, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }

        if self._variant == ModelVariant.UNION:
            inputs["controlnet_mode"] = torch.zeros(batch_size, dtype=torch.long)

        return inputs
