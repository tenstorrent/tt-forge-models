# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PixArt-Sigma model loader implementation
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src.model_utils import load_pipe, pixart_sigma_preprocessing


class ModelVariant(StrEnum):
    """Available PixArt-Sigma model variants."""

    XL_2_1024_MS = "XL-2-1024-MS"


class ModelLoader(ForgeModel):
    """PixArt-Sigma model loader implementation."""

    _VARIANTS = {
        ModelVariant.XL_2_1024_MS: ModelConfig(
            pretrained_model_name="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        )
    }

    DEFAULT_VARIANT = ModelVariant.XL_2_1024_MS

    prompt = "A small cactus with a happy face in the Sahara desert."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="PixArt-Sigma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.float32
        self.pipeline = load_pipe(
            self._variant_config.pretrained_model_name, dtype=dtype
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timestep,
            prompt_embeds,
            prompt_attention_mask,
        ) = pixart_sigma_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            prompt_attention_mask = prompt_attention_mask.to(dtype_override)

        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": prompt_embeds,
            "timestep": timestep,
            "added_cond_kwargs": {"resolution": None, "aspect_ratio": None},
            "encoder_attention_mask": prompt_attention_mask,
            "return_dict": False,
        }
