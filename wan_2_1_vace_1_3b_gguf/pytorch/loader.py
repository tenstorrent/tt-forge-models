#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 VACE 1.3B GGUF model loader implementation.

Loads GGUF-quantized Wan 2.1 VACE 1.3B transformers from
samuelchristlie/Wan2.1-VACE-1.3B-GGUF.

The Wan 2.1 VACE (Video All-in-one Creation Engine) model supports
versatile video creation and editing tasks. This loader uses GGUF-quantized
weights for reduced memory usage with the smaller 1.3B parameter variant.

Available variants:
- WAN21_VACE_1_3B_Q4_K_M: Q4_K_M quantization
- WAN21_VACE_1_3B_Q8_0: Q8_0 quantization
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

GGUF_REPO = "samuelchristlie/Wan2.1-VACE-1.3B-GGUF"
BASE_PIPELINE = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.1 VACE 1.3B GGUF variants."""

    WAN21_VACE_1_3B_Q4_K_M = "2.1_VACE_1_3B_Q4_K_M"
    WAN21_VACE_1_3B_Q8_0 = "2.1_VACE_1_3B_Q8_0"


_GGUF_FILES = {
    ModelVariant.WAN21_VACE_1_3B_Q4_K_M: "Wan2.1-VACE-1.3B-Q4_K_M.gguf",
    ModelVariant.WAN21_VACE_1_3B_Q8_0: "Wan2.1-VACE-1.3B-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.1 VACE 1.3B GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN21_VACE_1_3B_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN21_VACE_1_3B_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_VACE_1_3B_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_1_VACE_1_3B_GGUF",
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
        """Load the GGUF-quantized Wan 2.1 VACE 1.3B transformer.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer.
        Returns the transformer as a torch.nn.Module.
        """
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic tensor inputs for the Wan 2.1 VACE 1.3B transformer.

        Config: in_channels=16, text_dim=4096, patch_size=[1,2,2].
        hidden_states is 5D: (batch, channels, frames, height, width).
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = 1

        in_channels = 16
        text_dim = 4096
        txt_seq_len = 32

        # height and width must be multiples of patch_size [1,2,2]
        frames, height, width = 1, 4, 4

        hidden_states = torch.randn(
            batch_size, in_channels, frames, height, width, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        timestep = torch.tensor([500] * batch_size, dtype=torch.long)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
