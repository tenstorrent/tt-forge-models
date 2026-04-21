# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LCO-Embedding-Omni-7B-GGUF model loader implementation for embedding generation.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available LCO-Embedding-Omni-7B-GGUF model variants."""

    LCO_EMBEDDING_OMNI_7B_GGUF = "LCO-Embedding-Omni-7B-GGUF"


class ModelLoader(ForgeModel):
    """LCO-Embedding-Omni-7B-GGUF model loader for multimodal embedding generation."""

    _VARIANTS = {
        ModelVariant.LCO_EMBEDDING_OMNI_7B_GGUF: LLMModelConfig(
            pretrained_model_name="marksverdhei/LCO-Embedding-Omni-7B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LCO_EMBEDDING_OMNI_7B_GGUF

    GGUF_FILE = "LCO-Embedding-Omni-7B-Q4_K_M.gguf"

    sample_text = "Scaling language-centric omnimodal representation learning."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LCO-Embedding-Omni-7B-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, "logits"):
            hidden_states = outputs.logits
        else:
            hidden_states = outputs

        # LCO-Embedding-Omni uses last-token pooling to produce 3584-dim embeddings
        last_token_embedding = hidden_states[:, -1, :]
        return last_token_embedding

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits.flatten()
        if isinstance(fwd_output, (tuple, list)):
            return fwd_output[0].flatten()
        return fwd_output.flatten()

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
