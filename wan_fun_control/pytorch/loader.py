# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan Fun Control model loader for tt_forge_models.

Wan-Fun Control is a family of controllable video generation models from
alibaba-pai that support control conditions such as Canny, Depth, Pose, and
MLSD. Wan2.1-Fun-14B-Control uses a single WanTransformer3DModel with 36
input channels; Wan2.2-Fun-A14B-Control uses dual experts (high/low noise)
stored in subfolders, each with 52 input channels.

Repositories:
- https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control
- https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control
"""

from typing import Any, Optional

import torch

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
from .src.utils import (
    WAN21_TRANSFORMER_IN_CHANNELS,
    WAN22_TRANSFORMER_IN_CHANNELS,
    load_transformer,
    load_transformer_1_3b,
    load_transformer_inputs,
    load_transformer_inputs_1_3b,
)


class ModelVariant(StrEnum):
    """Available Wan Fun Control variants."""

    WAN21_FUN_14B_CONTROL = "2.1_Fun_14B_Control"
    WAN22_FUN_A14B_CONTROL = "2.2_Fun_A14B_Control"


# Per-variant loading metadata: the weights subfolder within the HF repo and
# the expected number of transformer input channels.
_VARIANT_LOAD_CONFIG = {
    ModelVariant.WAN21_FUN_14B_CONTROL: {
        "subfolder": None,
        "in_channels": WAN21_TRANSFORMER_IN_CHANNELS,
    },
    ModelVariant.WAN22_FUN_A14B_CONTROL: {
        "subfolder": "high_noise_model",
        "in_channels": WAN22_TRANSFORMER_IN_CHANNELS,
    },
}


class ModelLoader(ForgeModel):
    """
    Loader for the alibaba-pai Wan-Fun Control video generation family.

    Wan2.1-Fun-14B-Control stores the transformer at the repo root; the
    Wan2.2-Fun-A14B-Control checkpoint uses a dual-expert layout and the
    high-noise expert is loaded by default.
    """

    _VARIANTS = {
        ModelVariant.WAN21_FUN_14B_CONTROL: ModelConfig(
            pretrained_model_name="alibaba-pai/Wan2.1-Fun-14B-Control",
        ),
        ModelVariant.WAN22_FUN_A14B_CONTROL: ModelConfig(
            pretrained_model_name="alibaba-pai/Wan2.2-Fun-A14B-Control",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WAN21_FUN_14B_CONTROL

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="WanFunControl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        subfolder = _VARIANT_LOAD_CONFIG[self._variant]["subfolder"]
        return load_transformer(
            self._variant_config.pretrained_model_name,
            dtype,
            subfolder=subfolder,
        )

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        in_channels = _VARIANT_LOAD_CONFIG[self._variant]["in_channels"]
        return load_transformer_inputs(dtype, in_channels=in_channels)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
