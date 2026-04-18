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
        from accelerate import init_empty_weights
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
        from diffusers.loaders.single_file_utils import (
            convert_wan_transformer_to_diffusers,
        )
        from diffusers.models.model_loading_utils import (
            load_gguf_checkpoint,
            load_model_dict_into_meta,
        )
        from diffusers.quantizers import DiffusersAutoQuantizer
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]

        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        checkpoint = load_gguf_checkpoint(model_path)
        converted = convert_wan_transformer_to_diffusers(checkpoint)

        qconfig = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        hf_quantizer = DiffusersAutoQuantizer.from_config(qconfig)
        hf_quantizer.validate_environment()
        torch_dtype = hf_quantizer.update_torch_dtype(compute_dtype)

        config = WanTransformer3DModel.load_config(
            BASE_PIPELINE, subfolder="transformer"
        )
        with init_empty_weights():
            model = WanTransformer3DModel.from_config(config)

        hf_quantizer.preprocess_model(
            model=model, device_map=None, state_dict=converted, keep_in_fp32_modules=[]
        )

        load_model_dict_into_meta(
            model,
            converted,
            dtype=torch_dtype,
            device_map={"": torch.device("cpu")},
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=[],
            unexpected_keys=[],
        )

        hf_quantizer.postprocess_model(model)
        model.hf_quantizer = hf_quantizer
        model.eval()

        self.transformer = model
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
