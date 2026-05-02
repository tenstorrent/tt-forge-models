# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla-AI gemma-2-27b-it-llamafile model loader implementation for causal language modeling.

The mozilla-ai/gemma-2-27b-it-llamafile HuggingFace repo contains only llamafile
executables (GGUF models wrapped in a shell script), not standard HuggingFace model
files. This loader uses the equivalent GGUF model from bartowski/gemma-2-27b-it-GGUF
which provides the same Gemma 2 27B IT weights in GGUF Q4_K_M quantization.
"""
import torch
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
    """Available Mozilla-AI gemma-2-27b-it-llamafile model variants for causal language modeling."""

    GEMMA_2_27B_IT_LLAMAFILE = "gemma-2-27b-it-llamafile"


class ModelLoader(ForgeModel):
    """Mozilla-AI gemma-2-27b-it-llamafile model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_2_27B_IT_LLAMAFILE: LLMModelConfig(
            pretrained_model_name="bartowski/gemma-2-27b-it-GGUF",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_2_27B_IT_LLAMAFILE

    _GGUF_FILE = "gemma-2-27b-it-Q4_K_M.gguf"

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="gemma-2-27b-it-llamafile",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"gguf_file": self._GGUF_FILE}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": self._GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts = [text] * batch_size

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
