# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Fun InP GGUF model loader implementation for video generation
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

GGUF_REPO = "QuantStack/Wan2.2-Fun-A14B-InP-GGUF"
CONFIG_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 1
TRANSFORMER_HEIGHT = 32
TRANSFORMER_WIDTH = 32
TRANSFORMER_TEXT_SEQ_LEN = 64


class ModelVariant(StrEnum):
    """Available Wan 2.2 Fun InP GGUF model variants."""

    A14B_HIGHNOISE_Q4_K_M = "A14B_HighNoise_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.A14B_HIGHNOISE_Q4_K_M: "HighNoise/Wan2.2-Fun-A14B-InP_HighNoise-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Fun InP GGUF model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.A14B_HIGHNOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.A14B_HIGHNOISE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wan 2.2 Fun InP GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _ensure_gguf_available():
        import importlib.metadata
        import importlib.util

        import diffusers.utils.import_utils as _diffusers_import_utils

        if _diffusers_import_utils._gguf_available:
            return

        if importlib.util.find_spec("gguf") is None:
            return

        _diffusers_import_utils._gguf_available = True
        try:
            _diffusers_import_utils._gguf_version = importlib.metadata.version("gguf")
        except importlib.metadata.PackageNotFoundError:
            pass

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_gguf_available()

        from diffusers import WanTransformer3DModel
        from diffusers.models.model_loading_utils import load_gguf_checkpoint
        from diffusers.quantizers.gguf.utils import (
            GGUFParameter,
            dequantize_gguf_tensor,
        )
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        local_gguf_path = hf_hub_download(
            repo_id=GGUF_REPO,
            filename=gguf_file,
        )

        gguf_checkpoint = load_gguf_checkpoint(local_gguf_path)
        dequantized = {
            name: dequantize_gguf_tensor(param).to(compute_dtype)
            if isinstance(param, GGUFParameter)
            else param.to(compute_dtype)
            for name, param in gguf_checkpoint.items()
        }

        config = WanTransformer3DModel.load_config(
            CONFIG_REPO,
            subfolder="transformer",
        )
        self.transformer = WanTransformer3DModel.from_config(config)
        self.transformer.load_state_dict(dequantized, strict=False)
        self.transformer = self.transformer.to(compute_dtype)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        in_channels = config.in_channels
        hidden_states = torch.randn(
            batch_size,
            in_channels,
            TRANSFORMER_NUM_FRAMES,
            TRANSFORMER_HEIGHT,
            TRANSFORMER_WIDTH,
            dtype=dtype,
        )

        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)

        text_dim = config.text_dim
        encoder_hidden_states = torch.randn(
            batch_size, TRANSFORMER_TEXT_SEQ_LEN, text_dim, dtype=dtype
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
