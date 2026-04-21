# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS MLX model loader implementation for text-to-speech tasks.
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


class Qwen3TTSTalkerWrapper(nn.Module):
    """Wrapper around the Qwen3-TTS talker backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    (prefill mode) and produces codec logits for speech synthesis.
    """

    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds):
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        logits = self.codec_head(outputs.last_hidden_state)
        return logits


class ModelVariant(StrEnum):
    """Available Qwen3-TTS MLX model variants."""

    QWEN3_TTS_1_7B_VOICE_DESIGN_8BIT = "1.7B-VoiceDesign-8bit"


# Talker hidden size for the 1.7B variant.
_TALKER_HIDDEN_SIZE = {
    ModelVariant.QWEN3_TTS_1_7B_VOICE_DESIGN_8BIT: 2048,
}


class ModelLoader(ForgeModel):
    """Qwen3-TTS MLX model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_TTS_1_7B_VOICE_DESIGN_8BIT: ModelConfig(
            pretrained_model_name="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_TTS_1_7B_VOICE_DESIGN_8BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3-TTS MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import functools
        import transformers.utils.generic as _tug

        if not hasattr(_tug, "check_model_inputs"):
            # check_model_inputs was removed in transformers 5.x; provide a no-op shim
            def check_model_inputs():
                def decorator(func):
                    @functools.wraps(func)
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)

                    return wrapper

                return decorator

            _tug.check_model_inputs = check_model_inputs

        from qwen_tts.core.models import (
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
        )
        from transformers import AutoConfig, AutoModel

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

        # Load config and strip MLX quantization config (not compatible with transformers 5.x)
        model_name = self._variant_config.pretrained_model_name
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(cfg, "quantization_config"):
            del cfg.quantization_config

        # ignore_mismatched_sizes: the MLX-quantized weights use a different shape layout
        # than the standard transformers model; since this is compile-only, loading with
        # randomly initialised params for mismatched layers is acceptable.
        full_model = AutoModel.from_pretrained(
            model_name,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=dtype_override or torch.float32,
            ignore_mismatched_sizes=True,
        )
        model = Qwen3TTSTalkerWrapper(full_model.talker)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        hidden_size = _TALKER_HIDDEN_SIZE[self._variant]
        inputs_embeds = torch.randn(1, 32, hidden_size, dtype=dtype)
        return (inputs_embeds,)
