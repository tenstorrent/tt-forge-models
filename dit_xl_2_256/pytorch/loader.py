# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DiT-XL-2-256 model loader implementation for class-conditional image generation
"""
import torch
from diffusers import DiTTransformer2DModel
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available DiT-XL-2-256 model variants."""

    XL_2_256 = "XL-2-256"


class ModelLoader(ForgeModel):
    """DiT-XL-2-256 model loader implementation for class-conditional image generation."""

    _VARIANTS = {
        ModelVariant.XL_2_256: ModelConfig(
            pretrained_model_name="facebook/DiT-XL-2-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XL_2_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DiT-XL-2-256",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override
        load_kwargs |= kwargs

        self.transformer = DiTTransformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.transformer.config

        # Latent input: (B, C, H, W) where H,W = sample_size (32 for 256px model)
        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            config.sample_size,
            config.sample_size,
            dtype=dtype,
        )

        # Timestep
        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)

        # Class labels: ImageNet class indices (num_embeds_ada_norm = 1000 classes)
        class_labels = torch.randint(
            0, config.num_embeds_ada_norm, (batch_size,), dtype=torch.long
        )

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "class_labels": class_labels,
        }

        return inputs
