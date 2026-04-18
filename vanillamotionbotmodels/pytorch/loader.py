# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VanillaMotionBotModels Wan 2.2 I2V GGUF model loader implementation for video generation.

Loads Q8_0-quantized Wan 2.2 I2V 14B transformers from
bullerwins/Wan2.2-I2V-A14B-GGUF (public mirror of the gated
matadamovic/vanillamotionbotmodels checkpoints).
"""
import torch
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

GGUF_REPO = "bullerwins/Wan2.2-I2V-A14B-GGUF"
BASE_PIPELINE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


class ModelVariant(StrEnum):
    """Available VanillaMotionBotModels variants."""

    I2V_14B_HIGHNOISE_Q8_0 = "I2V_14B_HighNoise_Q8_0"
    I2V_14B_LOWNOISE_Q8_0 = "I2V_14B_LowNoise_Q8_0"


_GGUF_FILES = {
    ModelVariant.I2V_14B_HIGHNOISE_Q8_0: "wan2.2_i2v_high_noise_14B_Q8_0.gguf",
    ModelVariant.I2V_14B_LOWNOISE_Q8_0: "wan2.2_i2v_low_noise_14B_Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """VanillaMotionBotModels Wan 2.2 I2V GGUF model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.I2V_14B_HIGHNOISE_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.I2V_14B_LOWNOISE_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.I2V_14B_HIGHNOISE_Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VanillaMotionBotModels",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self.transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        num_frames = 1
        height = 32
        width = 32
        in_channels = config.in_channels

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=dtype
        )

        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)

        text_dim = config.text_dim
        seq_len = 64
        encoder_hidden_states = torch.randn(batch_size, seq_len, text_dim, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        return inputs
