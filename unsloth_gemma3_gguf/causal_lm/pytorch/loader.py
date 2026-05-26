# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Loader for the unsloth/gemma-3-1b-it-GGUF model.

The weights are distributed as GGUF files. transformers can ingest a GGUF
checkpoint via the ``gguf_file=`` argument; for the BF16 GGUF this performs
no quantization beyond the on-disk format, yielding the same architecture
as ``google/gemma-3-1b-it``.
"""

from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

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
from ....tools.utils import cast_input_to_type


@dataclass
class GGUFLLMModelConfig(LLMModelConfig):
    """LLM config extended with a GGUF filename inside the HF repo."""

    gguf_file: Optional[str] = None


class ModelVariant(StrEnum):
    """Available Unsloth Gemma3 GGUF variants for causal LM."""

    GEMMA_3_1B_IT_BF16 = "1b_it_bf16"


class ModelLoader(ForgeModel):
    """Loader for the Unsloth GGUF distribution of Gemma 3 1B IT."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_1B_IT_BF16: GGUFLLMModelConfig(
            pretrained_model_name="unsloth/gemma-3-1b-it-GGUF",
            max_length=16,
            gguf_file="gemma-3-1b-it-BF16.gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_1B_IT_BF16

    sample_text = "What is your favorite city?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Unsloth Gemma 3 GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        pretrained = self._variant_config.pretrained_model_name
        gguf_file = self._variant_config.gguf_file
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, gguf_file=gguf_file
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Gemma3 causal LM from a GGUF checkpoint."""
        pretrained = self._variant_config.pretrained_model_name
        gguf_file = self._variant_config.gguf_file

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"use_cache": False, "gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs)
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size: int = 1,
        prompt: Optional[str] = None,
    ):
        """Tokenize a small chat prompt and return [input_ids, attention_mask]."""
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [{"role": "user", "content": prompt or self.sample_text}]
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            input_text = messages[0]["content"]

        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = inputs["input_ids"].repeat_interleave(batch_size, dim=0)
        attn_mask = inputs["attention_mask"].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return [input_ids, attn_mask]
