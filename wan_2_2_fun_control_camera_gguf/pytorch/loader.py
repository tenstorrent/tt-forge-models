# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Fun A14B Control Camera GGUF model loader implementation for
text-to-video generation with camera control.

Loads GGUF-quantized Wan 2.2 Fun Control Camera transformers from
QuantStack/Wan2.2-Fun-A14B-Control-Camera-GGUF. The base model is
alibaba-pai/Wan2.2-Fun-A14B-Control-Camera, a fine-tune of
Wan-AI/Wan2.2-I2V-A14B with controllable camera movements.
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

GGUF_REPO = "QuantStack/Wan2.2-Fun-A14B-Control-Camera-GGUF"


class ModelVariant(StrEnum):
    """Available Wan 2.2 Fun Control Camera GGUF model variants."""

    A14B_HIGHNOISE_Q4_K_M = "A14B_HighNoise_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.A14B_HIGHNOISE_Q4_K_M: "HighNoise/Wan2.2-Fun-A14B-Control-Camera-HighNoise-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Fun Control Camera GGUF model loader for video generation tasks."""

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
            model="Wan 2.2 Fun Control Camera GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _ensure_gguf_available():
        """Patch diffusers gguf availability flags if gguf is installed but was
        not available when diffusers was first imported.

        load_gguf_checkpoint checks is_gguf_available() at call time, so we
        must ensure the flag is True before calling from_single_file.
        """
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
            repo_id=self._variant_config.pretrained_model_name,
            filename=gguf_file,
        )

        # load_gguf_checkpoint returns GGUFParameter (packed bytes) for quantized
        # tensors. Dequantize everything to plain float tensors so we can use
        # standard load_state_dict without a GGUFQuantizer (which would need
        # dispatch_model and fails when some params aren't in the GGUF file).
        gguf_checkpoint = load_gguf_checkpoint(local_gguf_path)
        dequantized = {
            name: dequantize_gguf_tensor(param).to(compute_dtype)
            if isinstance(param, GGUFParameter)
            else param.to(compute_dtype)
            for name, param in gguf_checkpoint.items()
        }

        # Create model from config only (no large weight download).
        # The GGUF encodes a Wan I2V 14B architecture so this config is compatible.
        config = WanTransformer3DModel.load_config(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
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

        # Video dimensions: (batch, channels, frames, height, width)
        num_frames = 1
        height = 32
        width = 32
        in_channels = config.in_channels

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)

        # Text encoder hidden states
        text_dim = config.text_dim
        seq_len = 64
        encoder_hidden_states = torch.randn(batch_size, seq_len, text_dim, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        return inputs
