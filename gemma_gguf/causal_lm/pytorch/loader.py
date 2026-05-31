# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma-2 GGUF model loader implementation for causal language modeling.

Loads pre-quantized GGUF checkpoints (from the lmstudio-community
gemma-2-9b-it-GGUF repository) through the HuggingFace transformers GGUF
integration, which dequantizes the weights into a standard torch nn.Module.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Gemma-2 GGUF model variants for causal LM."""

    GEMMA_2_9B_IT_Q4_K_M = "2_9b_it_q4_k_m"


class ModelLoader(ForgeModel):
    """Gemma-2 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_2_9B_IT_Q4_K_M: LLMModelConfig(
            pretrained_model_name="lmstudio-community/gemma-2-9b-it-GGUF",
            max_length=128,
        ),
    }

    # GGUF file within the repository to load for each variant.
    _GGUF_FILES = {
        ModelVariant.GEMMA_2_9B_IT_Q4_K_M: "gemma-2-9b-it-Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_2_9B_IT_Q4_K_M

    sample_text = "What is the capital of France, and what is it famous for?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Gemma-2 GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self) -> str:
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF checkpoint.

        Returns:
            The loaded tokenizer instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma-2 GGUF model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to cast the dequantized weights
                to. Defaults to torch.bfloat16 when not provided.

        Returns:
            torch.nn.Module: The Gemma-2 model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer()

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model_kwargs = {
            "gguf_file": self._gguf_file,
            "torch_dtype": dtype,
        }
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, prompt: Optional[str] = None):
        """Load and return sample inputs for the Gemma-2 GGUF model.

        Uses a single, fully-populated sequence (no padding) to keep the
        comparison against the reference numerically meaningful.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        input_prompt = prompt or self.sample_text
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            truncation=True,
        )

        if batch_size > 1:
            for key in inputs:
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        self.seq_len = inputs["input_ids"].shape[-1]
        return inputs

    def load_config(self):
        """Load and return the configuration for the Gemma-2 GGUF model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
