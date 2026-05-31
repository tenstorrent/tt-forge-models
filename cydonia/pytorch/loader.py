# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cydonia model loader implementation for causal language modeling.

Cydonia-24B-v2 (TheDrummer/Cydonia-24B-v2) is a finetune of
Mistral-Small-24B-Instruct-2501 (Mistral architecture, 40 layers).
This loader consumes the GGUF release published by bartowski; transformers
dequantizes the GGUF weights back into a standard torch model at load time.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Cydonia model variants."""

    CYDONIA_24B_V2_Q4_K_M = "24b_v2_q4_k_m"


class ModelLoader(ForgeModel):
    """Cydonia model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.CYDONIA_24B_V2_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/TheDrummer_Cydonia-24B-v2-GGUF",
            max_length=128,
        ),
    }

    # GGUF file (within the HuggingFace repo) backing each variant. transformers
    # dequantizes this single file into a Mistral causal-LM model.
    _GGUF_FILES = {
        ModelVariant.CYDONIA_24B_V2_Q4_K_M: "TheDrummer_Cydonia-24B-v2-Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CYDONIA_24B_V2_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Cydonia-24B-v2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self) -> str:
        """GGUF filename backing the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        The GGUF release does not ship a HF tokenizer; transformers reconstructs
        it from the GGUF metadata when ``gguf_file`` is provided.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
        )
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Cydonia model instance for this instance's variant.

        Args:
            dtype_override: Optional torch dtype to override the model's default.
                            If not provided, the model uses bfloat16.

        Returns:
            torch.nn.Module: The Cydonia model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # transformers dequantizes the GGUF file into a standard Mistral model.
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            gguf_file=self._gguf_file,
            torch_dtype=dtype,
            **kwargs,
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Cydonia model.

        Args:
            dtype_override: Unused for integer token inputs; kept for interface
                            compatibility.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        test_input = "How often does the letter r occur in Mistral?"

        inputs = self.tokenizer(test_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Keep the sequence short for bringup/compilation.
        max_len = 32
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        if batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
