# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Captain Eris Violet (abliterated) model loader implementation for causal language modeling.

This is a Mistral-Nemo (12B) architecture community model distributed as GGUF
(imatrix) quantized weights. The loader downloads a single GGUF file from the
HuggingFace repo and lets transformers dequantize it into a standard
``MistralForCausalLM`` instance, which can then run on CPU or Tenstorrent hardware.
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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Captain Eris Violet model variants for causal LM."""

    # Mradermacher imatrix GGUF quantization of
    # Nitral-Archive/Captain_Eris_Violet-0.420-abliterated. The Q4_K_M quant is a
    # standard K-quant that transformers can robustly dequantize; the loader
    # dequantizes it to the requested float dtype at load time.
    Q4_K_M_GGUF = "0.420_abliterated_i1_q4_k_m_gguf"


class ModelLoader(ForgeModel):
    """Captain Eris Violet model loader implementation for causal language modeling tasks."""

    # The GGUF file within the repo to load. transformers downloads only this file.
    _GGUF_FILE = "Captain_Eris_Violet-0.420-abliterated.i1-Q4_K_M.gguf"

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Captain_Eris_Violet-0.420-abliterated-i1-GGUF",
            max_length=32,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Q4_K_M_GGUF

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
            model="Captain_Eris_Violet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        The tokenizer is reconstructed from the GGUF metadata.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._GGUF_FILE
        )

        # Ensure a pad token exists for batched/padded inputs.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Captain Eris Violet model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype. If not
                provided, the model uses its default dtype (typically float32).

        Returns:
            torch.nn.Module: The model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._GGUF_FILE}
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
        """Load and return sample inputs for the model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # For causal LM, we provide input_ids and attention_mask padded to a
        # fixed length to keep the compiled shape small and static.
        target_len = self._variant_config.max_length
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            max_length=target_len,
            truncation=True,
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def load_config(self):
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the model.
        """
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._GGUF_FILE
        )

        return self.config
