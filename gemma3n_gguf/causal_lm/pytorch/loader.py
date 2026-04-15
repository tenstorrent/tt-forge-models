# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 3n GGUF model loader implementation for causal language modeling.
"""
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGemmaConverter
from transformers import AutoTokenizer, AutoConfig, Gemma3nForCausalLM
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

# Patch transformers GGUF support to recognize the gemma3n architecture.
# The gemma3n architecture is not yet in the upstream GGUF_SUPPORTED_ARCHITECTURES
# list but works with existing infrastructure once registered.
if "gemma3n" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
    _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("gemma3n")
if "gemma3n" not in GGUF_TO_FAST_CONVERTERS:
    GGUF_TO_FAST_CONVERTERS["gemma3n"] = GGUFGemmaConverter


class ModelVariant(StrEnum):
    """Available Gemma 3n GGUF model variants for causal language modeling."""

    GEMMA_3N_E4B_IT_Q4_K_M = "Gemma_3n_E4B_IT_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Gemma 3n GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3N_E4B_IT_Q4_K_M: LLMModelConfig(
            pretrained_model_name="NexaAI/gemma-3n",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3N_E4B_IT_Q4_K_M

    GGUF_FILE = "gemma-3n-E4B-it-Q4_K_M-full-vocab.gguf"

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Gemma 3n GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
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

        # Gemma3nForCausalLM expects Gemma3nTextConfig, not the composite
        # Gemma3nConfig. Load the full config and extract text_config so that
        # both normal loading and random-weight initialization work correctly.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        text_config = config.text_config
        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = text_config

        model = Gemma3nForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        input_prompt = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        input_text = self.tokenizer.apply_chat_template(
            input_prompt,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
