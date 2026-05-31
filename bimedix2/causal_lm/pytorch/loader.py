# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiMediX2 model loader implementation for causal language modeling.

BiMediX2-8B (MBZUAI/BiMediX2-8B-hf) is a bilingual medical large multimodal
model built on a Llama-3.1 text backbone with a CLIP vision encoder. This
loader targets the GGUF-quantized text backbone published by
``mradermacher/BiMediX2-8B-hf-i1-GGUF``. transformers dequantizes the GGUF
weights to a standard ``LlamaForCausalLM`` at load time, so the model is
exercised as a causal LM (the vision tower is not part of the GGUF export).
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import torch

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
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available BiMediX2 model variants for causal LM."""

    BIMEDIX2_8B_Q4_K_M = "8b_q4_k_m"


# Mapping of variant -> GGUF file within the quantized repo.
_GGUF_FILES = {
    ModelVariant.BIMEDIX2_8B_Q4_K_M: "BiMediX2-8B-hf.i1-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """BiMediX2 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BIMEDIX2_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/BiMediX2-8B-hf-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BIMEDIX2_8B_Q4_K_M

    # Sample text for causal LM (medical domain)
    sample_text = "What are the common symptoms of type 2 diabetes?"

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
            model="BiMediX2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self):
        """Return the GGUF filename for this instance's variant."""
        return _GGUF_FILES[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )

        # Ensure a pad token exists for batched / padded inputs.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the BiMediX2 model instance for this instance's variant.

        The GGUF-quantized weights are dequantized by transformers into a
        standard ``LlamaForCausalLM`` instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses the dtype produced by GGUF
                            dequantization (float32).

        Returns:
            torch.nn.Module: The BiMediX2 causal LM model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BiMediX2 model with this instance's settings.

        Args:
            dtype_override: Optional torch.dtype to override the default input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # For causal LM, we need both input_ids and attention_mask
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask to a fixed sequence length
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
