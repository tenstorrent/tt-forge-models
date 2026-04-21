# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Audio model loader implementation for audio-language tasks.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
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


class ModelVariant(StrEnum):
    """Available Qwen-Audio model variants."""

    QWEN_AUDIO_CHAT = "qwen_audio_chat"


class ModelLoader(ForgeModel):
    """Qwen-Audio model loader implementation for audio-language tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_AUDIO_CHAT: ModelConfig(
            pretrained_model_name="Qwen/Qwen-Audio-Chat",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_AUDIO_CHAT

    sample_audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac"
    sample_text = "what does the person say?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen-Audio",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        query = self.tokenizer.from_list_format(
            [
                {"audio": self.sample_audio_url},
                {"text": self.sample_text},
            ]
        )

        inputs = self.tokenizer(query, return_tensors="pt")
        return inputs
