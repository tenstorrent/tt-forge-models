# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-2512-Fun-Controlnet-Union model loader implementation.

Loads the ControlNet Union model for Qwen-Image-2512 from alibaba-pai.
This ControlNet supports multiple control conditions including Canny,
Depth, Pose, MLSD, HED, Scribble, and Gray.

Available variants:
- CONTROLNET_UNION: Original ControlNet Union weights
- CONTROLNET_UNION_2602: Updated ControlNet Union weights (2602 revision)
"""

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
from .src.model_utils import (
    create_dummy_inputs,
    load_controlnet_model,
)


class ModelVariant(StrEnum):
    """Available Qwen-Image-2512-Fun-Controlnet-Union model variants."""

    CONTROLNET_UNION = "ControlNet_Union"
    CONTROLNET_UNION_2602 = "ControlNet_Union_2602"


_VARIANT_FILENAMES = {
    ModelVariant.CONTROLNET_UNION: "Qwen-Image-2512-Fun-Controlnet-Union.safetensors",
    ModelVariant.CONTROLNET_UNION_2602: "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors",
}


class ModelLoader(ForgeModel):
    """Qwen-Image-2512-Fun-Controlnet-Union model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION: ModelConfig(
            pretrained_model_name="alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union",
        ),
        ModelVariant.CONTROLNET_UNION_2602: ModelConfig(
            pretrained_model_name="alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen-Image-2512-Fun-Controlnet-Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ControlNet Union model with loaded weights.
        """
        filename = _VARIANT_FILENAMES[self._variant]
        return load_controlnet_model(filename, dtype=dtype_override)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors for the ControlNet model.
        """
        dtype = dtype_override if dtype_override is not None else None
        return create_dummy_inputs(dtype=dtype) if dtype else create_dummy_inputs()
