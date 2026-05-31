# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-R1-Distill-Llama-8B-Debiased (GGUF) causal LM model loader implementation.

The upstream repo (``mradermacher/DeepSeek-R1-Distill-Llama-8B-Debiased-GGUF``)
ships only GGUF-quantized weight files and no ``config.json`` / tokenizer files.
``transformers`` is able to load such a repo by passing ``gguf_file=`` to the
``from_pretrained`` calls: the GGUF tensors are dequantized into a standard
``LlamaForCausalLM`` (float32) and the tokenizer / config are reconstructed from
the GGUF metadata. After loading, the model behaves like any other Llama causal
LM, so the device path is identical to the other Llama loaders.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available DeepSeek-R1-Distill-Llama-8B-Debiased GGUF variants."""

    # Q4_K_M is a widely-used, well-supported GGUF quantization and is
    # dequantized by transformers into standard float weights.
    LLAMA_8B_DEBIASED_Q4_K_M = "8B-Debiased-Q4_K_M"


class ModelLoader(ForgeModel):
    """DeepSeek-R1-Distill-Llama-8B-Debiased (GGUF) loader for causal LM tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_8B_DEBIASED_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/DeepSeek-R1-Distill-Llama-8B-Debiased-GGUF",
            max_length=128,
        ),
    }

    # GGUF weight file inside the repo to load for each variant.
    _GGUF_FILES = {
        ModelVariant.LLAMA_8B_DEBIASED_Q4_K_M: "DeepSeek-R1-Distill-Llama-8B-Debiased.Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_8B_DEBIASED_Q4_K_M

    sample_text = "Who are you?"

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
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DeepSeek Llama 8B Debiased GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the GGUF metadata.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # GGUF-reconstructed tokenizers may not define a pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepSeek Llama 8B Debiased GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its dequantized dtype (float32).

        Returns:
            torch.nn.Module: The Llama model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this variant's settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors (input_ids, attention_mask) suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask to a fixed length for the device.
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
