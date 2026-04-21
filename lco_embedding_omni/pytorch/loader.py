# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LCO-Embedding-Omni model loader implementation for multimodal embedding tasks.

LCO-Embedding-Omni is built on the Qwen2.5-Omni thinker component and produces
universal embeddings across text, image, audio, and video modalities. The
embedding is derived from the last hidden state of the final token and
L2-normalized.
"""
import torch
import torch.nn.functional as F
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
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
from .src.utils import last_token_pool


class ModelVariant(StrEnum):
    """Available LCO-Embedding-Omni model variants."""

    LCO_EMBEDDING_OMNI_7B = "7B"


class ModelLoader(ForgeModel):
    """LCO-Embedding-Omni model loader for multimodal embedding tasks."""

    _VARIANTS = {
        ModelVariant.LCO_EMBEDDING_OMNI_7B: ModelConfig(
            pretrained_model_name="LCO-Embedding/LCO-Embedding-Omni-7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LCO_EMBEDDING_OMNI_7B

    sample_texts = [
        "The capital of France is Paris.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LCO-Embedding-Omni",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.config.use_cache = False
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.sample_texts[0]},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0]

        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            embeddings = last_token_pool(hidden_states, attention_mask)
        else:
            embeddings = hidden_states[:, -1]

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()
