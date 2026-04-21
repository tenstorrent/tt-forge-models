#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 VACE 14B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 VACE transformers from the QuantStack
VACE GGUF repositories and returns the WanTransformer3DModel directly.

The Wan 2.1 VACE (Video All-in-one Creation Engine) model supports
versatile video creation and editing tasks including reference-to-video
generation. This loader uses GGUF-quantized weights for reduced memory
usage.

Available variants:
- WAN21_VACE_Q4_K_M: Q4_K_M quantization
- WAN21_VACE_Q8_0: Q8_0 quantization
- WAN21_FUSIONX_VACE_Q4_K_M: FusionX fine-tune, Q4_K_M quantization
- WAN21_FUSIONX_VACE_Q8_0: FusionX fine-tune, Q8_0 quantization
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download

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

VACE_GGUF_REPO = "QuantStack/Wan2.1_14B_VACE-GGUF"
FUSIONX_VACE_GGUF_REPO = "QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF"


class ModelVariant(StrEnum):
    """Available Wan 2.1 VACE 14B GGUF variants."""

    WAN21_VACE_Q4_K_M = "2.1_VACE_Q4_K_M"
    WAN21_VACE_Q8_0 = "2.1_VACE_Q8_0"
    WAN21_FUSIONX_VACE_Q4_K_M = "2.1_FusionX_VACE_Q4_K_M"
    WAN21_FUSIONX_VACE_Q8_0 = "2.1_FusionX_VACE_Q8_0"


_GGUF_FILES = {
    ModelVariant.WAN21_VACE_Q4_K_M: "Wan2.1_14B_VACE-Q4_K_M.gguf",
    ModelVariant.WAN21_VACE_Q8_0: "Wan2.1_14B_VACE-Q8_0.gguf",
    ModelVariant.WAN21_FUSIONX_VACE_Q4_K_M: "Wan2.1_T2V_14B_FusionX_VACE-Q4_K_M.gguf",
    ModelVariant.WAN21_FUSIONX_VACE_Q8_0: "Wan2.1_T2V_14B_FusionX_VACE-Q8_0.gguf",
}

_GGUF_REPOS = {
    ModelVariant.WAN21_VACE_Q4_K_M: VACE_GGUF_REPO,
    ModelVariant.WAN21_VACE_Q8_0: VACE_GGUF_REPO,
    ModelVariant.WAN21_FUSIONX_VACE_Q4_K_M: FUSIONX_VACE_GGUF_REPO,
    ModelVariant.WAN21_FUSIONX_VACE_Q8_0: FUSIONX_VACE_GGUF_REPO,
}


class ModelLoader(ForgeModel):
    """Wan 2.1 VACE 14B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN21_VACE_Q4_K_M: ModelConfig(
            pretrained_model_name=VACE_GGUF_REPO,
        ),
        ModelVariant.WAN21_VACE_Q8_0: ModelConfig(
            pretrained_model_name=VACE_GGUF_REPO,
        ),
        ModelVariant.WAN21_FUSIONX_VACE_Q4_K_M: ModelConfig(
            pretrained_model_name=FUSIONX_VACE_GGUF_REPO,
        ),
        ModelVariant.WAN21_FUSIONX_VACE_Q8_0: ModelConfig(
            pretrained_model_name=FUSIONX_VACE_GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_VACE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_1_VACE_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized Wan 2.1 VACE transformer.

        Returns the WanTransformer3DModel directly so the test framework
        receives a torch.nn.Module.
        """
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_repo = _GGUF_REPOS[self._variant]

        gguf_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_file)

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        self.transformer = WanTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        return self.transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Prepare dummy tensor inputs for the VACE transformer."""
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        in_channels = config.in_channels
        text_dim = config.text_dim

        hidden_states = torch.randn(batch_size, in_channels, 1, 32, 32, dtype=dtype)
        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)
        encoder_hidden_states = torch.randn(batch_size, 64, text_dim, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
