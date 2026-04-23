# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Audio model loader implementation for audio-language tasks.
"""

import transformers
import transformers.generation.utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

# transformers_stream_generator 0.0.5 imports beam search constraint classes that were
# removed in transformers>=4.45. Patch the missing symbols so it stays importable.
for _name in [
    "DisjunctiveConstraint",
    "BeamSearchScorer",
    "PhrasalConstraint",
    "ConstrainedBeamSearchScorer",
]:
    if not hasattr(transformers, _name):
        setattr(transformers, _name, object)
if not hasattr(transformers.generation.utils, "SampleOutput"):
    transformers.generation.utils.SampleOutput = (
        transformers.generation.utils.GenerateOutput
    )

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

    QWEN_AUDIO = "qwen_audio"
    QWEN_AUDIO_CHAT = "qwen_audio_chat"


class ModelLoader(ForgeModel):
    """Qwen-Audio model loader implementation for audio-language tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_AUDIO: ModelConfig(
            pretrained_model_name="Qwen/Qwen-Audio",
        ),
        ModelVariant.QWEN_AUDIO_CHAT: ModelConfig(
            pretrained_model_name="Qwen/Qwen-Audio-Chat",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_AUDIO

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
            task=ModelTask.MM_CONDITIONAL_GENERATION,
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
                {
                    "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac",
                },
                {"text": "what does the person say?"},
            ]
        )

        audio_info = self.tokenizer.process_audio(query)
        inputs = self.tokenizer(query, return_tensors="pt", audio_info=audio_info)
        inputs["audio_info"] = audio_info
        return inputs
