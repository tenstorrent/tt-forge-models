# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit model loader implementation for text-to-speech tasks.
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
    """Available mlx-community Qwen3-TTS-12Hz-0.6B-Base-8bit model variants."""

    QWEN3_TTS_12HZ_0_6B_BASE_8BIT = "12Hz-0.6B-Base-8bit"


# Talker hidden size for the 0.6B base variant.
_TALKER_HIDDEN_SIZE = 1024


class ModelLoader(ForgeModel):
    """mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit model loader for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_TTS_12HZ_0_6B_BASE_8BIT: ModelConfig(
            pretrained_model_name="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_TTS_12HZ_0_6B_BASE_8BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mlx-community Qwen3-TTS-12Hz-0.6B-Base-8bit",
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

        # The mlx-community checkpoint uses MLX 8-bit affine quantization (uint32-packed
        # weights) which is incompatible with the standard transformers>=5.x loading
        # pipeline. Since we only need the architecture for compilation, we initialize
        # the talker from config with random weights instead of loading the checkpoint.
        config = Qwen3TTSConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        full_model = Qwen3TTSForConditionalGeneration(config)
        dtype = dtype_override or torch.float32
        full_model = full_model.to(dtype)
        model = Qwen3TTSTalkerWrapper(full_model.talker)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        inputs_embeds = torch.randn(1, 32, _TALKER_HIDDEN_SIZE, dtype=dtype)
        return (inputs_embeds,)
