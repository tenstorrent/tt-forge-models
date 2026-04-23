# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VanillaMotionBotModels Wan 2.2 I2V GGUF model loader implementation for video generation
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

REPO_ID = "bullerwins/Wan2.2-I2V-A14B-GGUF"
CONFIG_REPO = "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers"

_GGUF_FILES = {
    "I2V_14B_HIGHNOISE_Q8_0": "wan2.2_i2v_high_noise_14B_Q8_0.gguf",
    "I2V_14B_LOWNOISE_Q8_0": "wan2.2_i2v_low_noise_14B_Q8_0.gguf",
}


class ModelVariant(StrEnum):
    """Available VanillaMotionBotModels variants."""

    I2V_14B_HIGHNOISE_Q8_0 = "I2V_14B_HighNoise_Q8_0"
    I2V_14B_LOWNOISE_Q8_0 = "I2V_14B_LowNoise_Q8_0"


class ModelLoader(ForgeModel):
    """VanillaMotionBotModels Wan 2.2 I2V GGUF model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.I2V_14B_HIGHNOISE_Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.I2V_14B_LOWNOISE_Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.I2V_14B_HIGHNOISE_Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

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
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant.name]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{REPO_ID}/{gguf_file}",
            quantization_config=quantization_config,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=compute_dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self._transformer.config

        in_channels = config.in_channels
        text_dim = config.text_dim

        num_frames = 2
        height = 4
        width = 4

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=dtype
        )
        timestep = torch.tensor([500] * batch_size, dtype=torch.long)
        encoder_hidden_states = torch.randn(batch_size, 32, text_dim, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }
