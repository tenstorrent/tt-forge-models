# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Marigold Normals model loader implementation for monocular surface normal estimation.
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
from .src.model_utils import load_pipe, marigold_normals_preprocessing


class ModelVariant(StrEnum):
    """Available Marigold Normals model variants."""

    V0_1 = "v0.1"


class ModelLoader(ForgeModel):
    """Marigold Normals model loader implementation."""

    _VARIANTS = {
        ModelVariant.V0_1: ModelConfig(
            pretrained_model_name="prs-eth/marigold-normals-v0-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MarigoldNormals",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name, dtype=dtype_override)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        sample, timestep, encoder_hidden_states = marigold_normals_preprocessing(
            self.pipeline
        )

        if dtype_override:
            sample = sample.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
