# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Kyutai STT model loader implementation for speech recognition (ASR) using PyTorch.
"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Kyutai STT PyTorch speech recognition model variants."""

    STT_2_6B_EN = "STT_2.6B_EN"


class ModelLoader(ForgeModel):
    """Kyutai STT model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.STT_2_6B_EN: ModelConfig(
            pretrained_model_name="kyutai/stt-2.6b-en-trfs",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STT_2_6B_EN

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="KyutaiSTT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import KyutaiSpeechToTextForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import torch

        config = self._variant_config
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(config.pretrained_model_name)

        seq_len = model_config.sliding_window
        num_channels = model_config.num_codebooks + 1
        input_ids = torch.zeros(1, seq_len, num_channels, dtype=torch.long)
        input_ids[:, :, 0] = torch.randint(0, model_config.vocab_size, (1, seq_len))
        for i in range(1, num_channels):
            input_ids[:, :, i] = torch.randint(
                0, model_config.codebook_vocab_size, (1, seq_len)
            )

        return {"input_ids": input_ids}
