# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
React Native ExecuTorch BK-SDM-Tiny text-to-image model loader implementation.

Note: The original model (software-mansion/react-native-executorch-bk-sdm-tiny) is in
ExecuTorch .pte format intended for mobile inference. Since ExecuTorch format is not
compatible with PyTorch, this loader uses the upstream base model
(vivym/bk-sdm-tiny-vpred) via StableDiffusionPipeline.
"""

import torch
from typing import Optional

from diffusers import StableDiffusionPipeline

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
    """Available React Native ExecuTorch BK-SDM-Tiny model variants."""

    BK_SDM_TINY = "bk-sdm-tiny"


class ModelLoader(ForgeModel):
    """React Native ExecuTorch BK-SDM-Tiny text-to-image model loader implementation."""

    _VARIANTS = {
        ModelVariant.BK_SDM_TINY: ModelConfig(
            pretrained_model_name="vivym/bk-sdm-tiny-vpred",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BK_SDM_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="React Native ExecuTorch BK-SDM-Tiny",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BK-SDM-Tiny pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionPipeline: The pre-trained BK-SDM-Tiny pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for BK-SDM-Tiny.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "a tropical bird sitting on a branch",
        ] * batch_size
        return prompt
