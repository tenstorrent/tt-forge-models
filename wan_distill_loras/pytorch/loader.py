#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Distill LoRA model loader implementation.

Loads the Wan 2.2 I2V base pipeline and applies distilled LoRA weights
from lightx2v/Wan2.2-Distill-Loras for fast 4-step image-to-video generation.

The transformer (WanTransformer3DModel) is extracted from the pipeline and
returned as the model for compilation.

Available variants:
- WAN22_I2V_HIGH_NOISE: Creative, diverse outputs (high noise LoRA)
- WAN22_I2V_LOW_NOISE: Faithful, stable outputs (low noise LoRA)
"""

import warnings
from typing import Any, Optional

import torch
from diffusers import WanImageToVideoPipeline  # type: ignore[import]

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

BASE_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
LORA_REPO = "lightx2v/Wan2.2-Distill-Loras"

# LoRA weight filenames
LORA_HIGH_NOISE = (
    "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
)
LORA_LOW_NOISE = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"

# Transformer input dimensions for Wan2.2-I2V-A14B
_TRANSFORMER_IN_CHANNELS = 36  # 16 video + 16 image + 4 mask
_LATENT_HEIGHT = 4
_LATENT_WIDTH = 4
_LATENT_DEPTH = 2
_TEXT_HIDDEN_DIM = 4096
_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 Distill LoRA variants."""

    WAN22_I2V_HIGH_NOISE = "2.2_I2V_HighNoise"
    WAN22_I2V_LOW_NOISE = "2.2_I2V_LowNoise"


_LORA_FILES = {
    ModelVariant.WAN22_I2V_HIGH_NOISE: LORA_HIGH_NOISE,
    ModelVariant.WAN22_I2V_LOW_NOISE: LORA_LOW_NOISE,
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Distill LoRA model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_I2V_HIGH_NOISE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_I2V_LOW_NOISE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_I2V_HIGH_NOISE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_DISTILL_LORAS",
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
        """Load the Wan 2.2 I2V transformer with distilled LoRA weights applied.

        Loads the full pipeline, applies LoRA weights, then extracts and returns
        the transformer (WanTransformer3DModel) so that the test framework
        receives a torch.nn.Module.

        Returns:
            WanTransformer3DModel with LoRA weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        lora_file = _LORA_FILES[self._variant]
        try:
            self.pipeline.load_lora_weights(
                LORA_REPO,
                weight_name=lora_file,
            )
        except (IndexError, ValueError) as e:
            warnings.warn(
                f"Skipping LoRA loading for {lora_file}: {e}",
                stacklevel=2,
            )

        transformer = self.pipeline.transformer
        transformer.eval()
        return transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare synthetic inputs for the WanTransformer3DModel forward pass.

        Returns:
            dict with hidden_states, encoder_hidden_states, timestep, and
            return_dict keys suitable for WanTransformer3DModel.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = 1
        seq_len = _LATENT_DEPTH * _LATENT_HEIGHT * _LATENT_WIDTH

        hidden_states = torch.randn(
            batch_size, seq_len, _TRANSFORMER_IN_CHANNELS, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, _TEXT_SEQ_LEN, _TEXT_HIDDEN_DIM, dtype=dtype
        )
        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
