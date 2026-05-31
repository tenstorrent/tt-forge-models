# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-4o-distilled Llama-3.1-8B-Instruct (PaperWitch heresy) GGUF loader for
causal language modeling.

This model is distributed only in GGUF format (a quantized weight format used by
llama.cpp). The underlying architecture is standard Llama-3.1-8B, so it is loaded
through transformers' GGUF support: ``from_pretrained(repo, gguf_file=...)``
dequantizes the GGUF tensors back into a regular ``LlamaForCausalLM`` torch module.
The embedded GGUF tokenizer is loaded the same way.
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
    """Available GPT-4o-distil Llama GGUF variants for causal LM."""

    # gpt-4o distilled Llama-3.1-8B-Instruct, PaperWitch "heresy" fine-tune,
    # Q4_K_M GGUF quantization (dequantized on load).
    LLAMA_3_1_8B_PAPERWITCH_Q4_K_M = "3.1_8b_instruct_paperwitch_q4_k_m_gguf"


class ModelLoader(ForgeModel):
    """GPT-4o-distil Llama GGUF loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LLAMA_3_1_8B_PAPERWITCH_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/gpt-4o-distil-Llama-3.1-8B-Instruct-PaperWitch-heresy-GGUF",
            max_length=128,
        ),
    }

    # GGUF filename within the repo for each variant. The repo ships multiple
    # quantizations; this selects the specific file transformers should dequantize.
    _GGUF_FILES = {
        ModelVariant.LLAMA_3_1_8B_PAPERWITCH_Q4_K_M: "gpt-4o-distil-Llama-3.1-8B-Instruct-PaperWitch-heresy.Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_8B_PAPERWITCH_Q4_K_M

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
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="gpt4o_distil_llama",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Ensure a pad token exists for Llama-style tokenizers.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the model instance, dequantized from GGUF.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, transformers dequantizes GGUF to float32.

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": gguf_file}
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
        """Load and return sample inputs for the model with this instance's settings.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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

        # Pad input_ids and attention_mask to a static target length
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
