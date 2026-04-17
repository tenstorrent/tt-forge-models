# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS-Tokenizer model loader implementation for audio feature extraction.
"""
import torch
import torch.nn as nn
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


class Qwen3TTSTokenizerEncoderWrapper(nn.Module):
    """Wrapper around the Qwen3-TTS-Tokenizer's MimiEncoder.

    Takes raw mono audio waveform (batch, 1, samples) and produces
    a feature embedding via the convolutional encoder layers.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio):
        return self.encoder(audio)


class ModelVariant(StrEnum):
    """Available Qwen3-TTS-Tokenizer model variants."""

    QWEN3_TTS_TOKENIZER_12HZ = "12Hz"


class ModelLoader(ForgeModel):
    """Qwen3-TTS-Tokenizer model loader for audio feature extraction."""

    @staticmethod
    def _patch_transformers_compat():
        """Patch transformers 5.x for qwen_tts compatibility."""
        from functools import wraps

        import transformers.utils.generic as generic_utils

        if not hasattr(generic_utils, "check_model_inputs"):

            def check_model_inputs(tie_last_hidden_states=True):
                def wrapped_fn(func):
                    @wraps(func)
                    def wrapper(self, *args, **kwargs):
                        return func(self, *args, **kwargs)

                    return wrapper

                return wrapped_fn

            generic_utils.check_model_inputs = check_model_inputs

        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" not in ROPE_INIT_FUNCTIONS:

            def _compute_default_rope_parameters(
                config=None, device=None, seq_len=None
            ):
                base = config.rope_theta
                partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads
                )
                dim = int(head_dim * partial_rotary_factor)
                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dim, 2, dtype=torch.int64).to(
                            device=device, dtype=torch.float
                        )
                        / dim
                    )
                )
                return inv_freq, 1.0

            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    _VARIANTS = {
        ModelVariant.QWEN3_TTS_TOKENIZER_12HZ: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_TTS_TOKENIZER_12HZ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3-TTS-Tokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self._patch_transformers_compat()
        from qwen_tts import Qwen3TTSTokenizer

        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        # Use the inner MimiEncoder (convolutional layers only)
        model = Qwen3TTSTokenizerEncoderWrapper(tokenizer.model.encoder.encoder)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # MimiEncoder expects raw mono audio: (batch, channels=1, samples)
        # 24kHz sample rate, use a multiple of the downsample rate (1920)
        audio = torch.randn(1, 1, 19200, dtype=dtype)
        return (audio,)
