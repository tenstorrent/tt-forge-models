# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Curie-7B-v1 (GGUF) model loader implementation for causal language modeling.

Curie-7B-v1 is a Mistral-7B based Polish language model. This loader consumes
the GGUF (quantized) distribution published by RichardErkhov; transformers
dequantizes the GGUF weights back to a standard MistralForCausalLM at load time.
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
from ....tools.utils import (
    cast_input_to_type,
)


class ModelVariant(StrEnum):
    """Available Curie-7B-v1 GGUF model variants for causal LM."""

    CURIE_7B_V1_Q4_K_M = "7b_v1_q4_k_m"


class ModelLoader(ForgeModel):
    """Curie-7B-v1 GGUF model loader for causal language modeling tasks."""

    # GGUF file published inside the repo for each variant.
    _GGUF_FILES = {
        ModelVariant.CURIE_7B_V1_Q4_K_M: "Curie-7B-v1.Q4_K_M.gguf",
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.CURIE_7B_V1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="RichardErkhov/szymonrucinski_-_Curie-7B-v1-gguf",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CURIE_7B_V1_Q4_K_M

    # Sample text for causal LM (Curie-7B-v1 is a Polish language model)
    sample_text = "Jak się masz dzisiaj?"

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
            model="Curie",
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

        # Ensure a pad token exists (mirrors Mistral/Llama convention).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Curie-7B-v1 model instance for this variant.

        The GGUF weights are dequantized by transformers into a standard
        MistralForCausalLM module.

        Args:
            dtype_override: Optional torch.dtype to override the loaded dtype.
                            If not provided, the dequantized default dtype is used.
        Returns:
            torch.nn.Module: The Curie-7B-v1 model instance for causal LM.
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
        """Load and return sample inputs for the Curie-7B-v1 model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
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

        # Use the natural (unpadded) sequence length. Padding tokens produce
        # ill-defined logits that diverge between host golden and device,
        # which would hurt PCC without exercising additional model behaviour.
        self.seq_len = inputs["input_ids"].shape[1]
        return inputs
