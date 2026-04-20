# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan Fun InP model loader for tt_forge_models.

Wan2.1-Fun-V1.1-1.3B-InP is a text/image-to-video diffusion transformer from
alibaba-pai. It uses a WanTransformer3DModel with 36 input channels (16
latent + 16 image latent + 4 mask/extra conditioning) for start/end image
prediction.

Repository: https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP
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
    load_transformer_inputs,
)


class ModelVariant(StrEnum):
    """Available Wan Fun InP variants."""

    WAN21_FUN_V1_1_1_3B_INP = "2.1_Fun_V1.1_1.3B_InP"


class ModelLoader(ForgeModel):
    """
    Loader for alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP video generation model.

    Loads the WanTransformer3DModel with 36 input channels for I2V
    conditioning. The repo stores a minimal WanModel config that diffusers
    cannot load directly, so weights are loaded via from_single_file using
    the standard Wan 2.1 T2V 1.3B transformer config as a reference.
    """

    _VARIANTS = {
        ModelVariant.WAN21_FUN_V1_1_1_3B_INP: ModelConfig(
            pretrained_model_name="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WAN21_FUN_V1_1_1_3B_INP

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
            model="WanFunInP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return load_transformer(self._variant_config.pretrained_model_name, dtype)

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return load_transformer_inputs(dtype)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
