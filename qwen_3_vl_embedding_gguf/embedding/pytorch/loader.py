# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL Embedding GGUF model loader implementation for multimodal embedding tasks.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
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
from .src.utils import last_token_pool


class ModelVariant(StrEnum):
    """Available Qwen 3 VL Embedding GGUF model variants."""

    QWEN_3_VL_EMBEDDING_8B_Q4_K_M = "Embedding_8B_Q4_K_M"


class ModelLoader(ForgeModel):
    """Qwen 3 VL Embedding GGUF model loader implementation for multimodal embedding tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_EMBEDDING_8B_Q4_K_M: ModelConfig(
            pretrained_model_name="dam2452/Qwen3-VL-Embedding-8B-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_EMBEDDING_8B_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.QWEN_3_VL_EMBEDDING_8B_Q4_K_M: "Qwen3-VL-Embedding-8B-Q4_K_M.gguf",
    }

    sample_queries = [
        "A woman and a dog playing on the beach",
        "A mountain landscape at sunset",
    ]
    sample_documents = [
        "Two friends enjoying a day at the seaside with their pet.",
        "Snow-capped peaks glowing with golden evening light.",
    ]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3 VL Embedding GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._GGUF_FILES[self._variant]
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, max_length=128):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_texts = self.sample_queries + self.sample_documents

        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0]

        embeddings = last_token_pool(hidden_states, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        num_queries = len(self.sample_queries)
        scores = embeddings[:num_queries] @ embeddings[num_queries:].T

        return scores.tolist()

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
