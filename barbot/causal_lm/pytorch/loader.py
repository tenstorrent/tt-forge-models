# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Barbot model loader implementation for causal language modeling.

Barbot-8B-v1 is a Llama-3 architecture finetune distributed only as GGUF
(llama.cpp) quantized weights. This loader pulls the imatrix Q4_K_M GGUF and
lets transformers dequantize it back into a standard ``LlamaForCausalLM`` so it
can be compiled and run on Tenstorrent hardware like any other Llama model.
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
    """Available Barbot model variants for causal LM."""

    # imatrix Q4_K_M GGUF quantization of Barbot-8B-v1 (Llama-3 architecture).
    BARBOT_8B_V1_I1_Q4_K_M = "8b_v1_i1_q4_k_m"


# GGUF file inside the HuggingFace repo to dequantize for each variant.
_GGUF_FILES = {
    ModelVariant.BARBOT_8B_V1_I1_Q4_K_M: "Barbot-8B-v1.i1-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Barbot model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BARBOT_8B_V1_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Barbot-8B-v1-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BARBOT_8B_V1_I1_Q4_K_M

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
            model="Barbot",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self):
        """Return the GGUF filename for the current variant."""
        return _GGUF_FILES[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        The tokenizer is embedded in the GGUF file, so it is loaded by pointing
        ``AutoTokenizer`` at the repo together with the ``gguf_file`` name.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )

        # Set pad token to eos token for Llama-family models
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Barbot model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.
                            If not provided, the dequantized GGUF weights default
                            to float32.

        Returns:
            torch.nn.Module: The Barbot (Llama) model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Barbot model.

        Args:
            dtype_override: Optional torch.dtype to override input dtype. Integer
                            inputs (input_ids) are left untouched.
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

        # Only convert dtype if explicitly requested (no-op for integer ids)
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask to a fixed length
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
