# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan Fun Control model loader for tt_forge_models.

Wan2.1-Fun-*-Control are controllable video generation models from alibaba-pai
that support control conditions such as Canny, Depth, Pose, and MLSD. Both
variants use a WanTransformer3DModel backbone with extra input channels for
control/mask conditioning.

Repositories:
- https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control
- https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control
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
    load_transformer,
    load_transformer_1_3b,
    load_transformer_inputs,
    load_transformer_inputs_1_3b,
)


class ModelVariant(StrEnum):
    """Available Wan Fun Control variants."""

    WAN21_FUN_14B_CONTROL = "2.1_Fun_14B_Control"
    WAN21_FUN_1_3B_CONTROL = "2.1_Fun_1.3B_Control"


class ModelLoader(ForgeModel):
    """
    Loader for the alibaba-pai Wan 2.1 Fun Control video generation models.

    The 14B variant stores a diffusers-compatible config at the repo root and
    is loaded directly via ``WanTransformer3DModel.from_pretrained``. The 1.3B
    variant ships a minimal ``WanModel`` config, so we load its safetensors
    through ``from_single_file`` using the standard Wan 2.1 T2V 1.3B config as
    a reference.
    """

    _VARIANTS = {
        ModelVariant.WAN21_FUN_14B_CONTROL: ModelConfig(
            pretrained_model_name="alibaba-pai/Wan2.1-Fun-14B-Control",
        ),
        ModelVariant.WAN21_FUN_1_3B_CONTROL: ModelConfig(
            pretrained_model_name="alibaba-pai/Wan2.1-Fun-1.3B-Control",
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
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self._variant == ModelVariant.WAN21_FUN_1_3B_CONTROL:
            return load_transformer_1_3b(pretrained_model_name, dtype)
        return load_transformer(pretrained_model_name, dtype)

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._variant == ModelVariant.WAN21_FUN_1_3B_CONTROL:
            return load_transformer_inputs_1_3b(dtype)
        return load_transformer_inputs(dtype)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
