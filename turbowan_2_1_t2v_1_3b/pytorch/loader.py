# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TurboWan2.1-T2V-1.3B-Diffusers text-to-video model loader implementation.

TurboWan2.1-T2V-1.3B-Diffusers is a distilled/turbo variant of the
Wan-AI/Wan2.1-T2V-1.3B text-to-video diffusion model, finetuned from
TurboDiffusion/TurboWan2.1-T2V-1.3B-480P for fast few-step inference.
"""

from typing import Any, Dict, Optional

import diffusers  # type: ignore[import]
import torch
from diffusers import DiffusionPipeline, WanPipeline  # type: ignore[import]

# WanDMDPipeline is not yet in a released version of diffusers; use WanPipeline as a stand-in.
if not hasattr(diffusers, "WanDMDPipeline"):
    diffusers.WanDMDPipeline = WanPipeline

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

# Wan 2.1 1.3B transformer config: in_channels=16, text_dim=4096, patch_size=[1,2,2]
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available TurboWan2.1-T2V-1.3B model variants."""

    TURBOWAN_2_1_T2V_1_3B = "2.1_T2V_1.3B"


class ModelLoader(ForgeModel):
    """TurboWan2.1-T2V-1.3B-Diffusers text-to-video model loader implementation."""

    _VARIANTS = {
        ModelVariant.TURBOWAN_2_1_T2V_1_3B: ModelConfig(
            pretrained_model_name="IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TURBOWAN_2_1_T2V_1_3B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TurboWan2.1-T2V-1.3B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self, dtype_override: Optional[torch.dtype] = None
    ) -> DiffusionPipeline:
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=(
                dtype_override if dtype_override is not None else torch.bfloat16
            ),
        )
        return self.pipeline

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the WanTransformer3DModel from the TurboWan pipeline.

        Args:
            dtype_override: Optional torch dtype to instantiate the pipeline with.

        Returns:
            WanTransformer3DModel: The transformer component.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline.transformer

    def _load_transformer_inputs(self, dtype: torch.dtype) -> Dict[str, Any]:
        """Prepare inputs for the WanTransformer3DModel forward pass."""
        config = self.pipeline.transformer.config
        return {
            "hidden_states": torch.randn(
                1,
                config.in_channels,
                TRANSFORMER_NUM_FRAMES,
                TRANSFORMER_HEIGHT,
                TRANSFORMER_WIDTH,
                dtype=dtype,
            ),
            "encoder_hidden_states": torch.randn(
                1,
                TRANSFORMER_TEXT_SEQ_LEN,
                config.text_dim,
                dtype=dtype,
            ),
            "timestep": torch.tensor([500], dtype=torch.long),
            "return_dict": False,
        }

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the WanTransformer3DModel.

        Returns:
            dict: Tensor inputs for the transformer forward pass.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        return self._load_transformer_inputs(dtype)
