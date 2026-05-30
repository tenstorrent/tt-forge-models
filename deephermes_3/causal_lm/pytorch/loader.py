# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepHermes-3 (Llama-3 architecture) GGUF model loader for causal language modeling.

The weights are sourced from a GGUF repository on HuggingFace. transformers
dequantizes the GGUF tensors into a standard Llama model at load time via the
``gguf_file`` argument to ``from_pretrained``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type


@dataclass
class GGUFModelConfig(LLMModelConfig):
    """LLM config that also records which GGUF file to dequantize."""

    gguf_file: Optional[str] = None


class ModelVariant(StrEnum):
    """Available DeepHermes-3 GGUF variants for causal LM."""

    LLAMA_3_3B_Q4_K_M = "deephermes_3_llama_3_3b_q4_k_m"


class ModelLoader(ForgeModel):
    """DeepHermes-3 GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_3B_Q4_K_M: GGUFModelConfig(
            pretrained_model_name="bartowski/NousResearch_DeepHermes-3-Llama-3-3B-Preview-GGUF",
            gguf_file="NousResearch_DeepHermes-3-Llama-3-3B-Preview-Q4_K_M.gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_3B_Q4_K_M

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="DeepHermes-3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._variant_config.gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Llama tokenizers have no pad token by default; reuse eos.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepHermes-3 model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the dequantized GGUF default (float32) is used.

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._variant_config.gguf_file

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DeepHermes-3 model.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Tokenize without padding so the device-vs-CPU PCC reflects real token
        # positions rather than meaningless padding logits, which diverge heavily
        # in bfloat16 and would otherwise dominate the correlation.
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        self.seq_len = inputs["input_ids"].shape[-1]
        return inputs

    def load_config(self):
        """Load and return the configuration for the DeepHermes-3 model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._variant_config.gguf_file,
        )
        return self.config
