# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit model loader implementation
for text-to-speech tasks.

Note: The mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit model is an
MLX-quantized variant of Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice. Since MLX
models cannot be loaded directly with transformers, this loader uses the
base Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice model.
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
    """Available mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit model variants."""

    QWEN3_TTS_1_7B_CUSTOM_VOICE_8BIT = "1.7B-CustomVoice-8bit"


class ModelLoader(ForgeModel):
    """mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit loader for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_TTS_1_7B_CUSTOM_VOICE_8BIT: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_TTS_1_7B_CUSTOM_VOICE_8BIT

    # Talker hidden size for the 1.7B CustomVoice model.
    _TALKER_HIDDEN_SIZE = 2048

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from qwen_tts.core.models import (
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
        )
        from transformers import AutoConfig, AutoModel

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            dtype=dtype_override or torch.float32,
        )
        model = Qwen3TTSTalkerWrapper(full_model.talker)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        inputs_embeds = torch.randn(1, 32, self._TALKER_HIDDEN_SIZE, dtype=dtype)
        return (inputs_embeds,)
