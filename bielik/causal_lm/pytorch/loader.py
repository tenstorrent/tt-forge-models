# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bielik model loader implementation for causal language modeling.

Bielik-11B-v3.0-Instruct is a Polish LLM built on the Mistral/Llama
architecture. Only GGUF weights are published openly (the base safetensors
repo is gated), so both the model and the tokenizer are loaded from a GGUF
file via transformers' GGUF integration, which dequantizes the weights into a
standard ``LlamaForCausalLM`` at load time.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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


class ModelVariant(StrEnum):
    """Available Bielik model variants for causal LM."""

    BIELIK_11B_V3_0_INSTRUCT = "11B_v3.0_Instruct"


class ModelLoader(ForgeModel):
    """Bielik model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BIELIK_11B_V3_0_INSTRUCT: LLMModelConfig(
            pretrained_model_name="speakleash/Bielik-11B-v3.0-Instruct-GGUF",
            max_length=32,
        ),
    }

    # The base safetensors repo is gated; weights and tokenizer are loaded from
    # the published GGUF file. transformers dequantizes the GGUF to a standard
    # LlamaForCausalLM. Q4_K_M is the smallest published quant.
    _GGUF_FILES = {
        ModelVariant.BIELIK_11B_V3_0_INSTRUCT: "Bielik-11B-v3.0-Instruct.Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BIELIK_11B_V3_0_INSTRUCT

    # Sample text for causal LM
    sample_text = "Kim był Mikołaj Kopernik?"

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
        return ModelInfo(
            model="Bielik",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Ensure a pad token is available for batched/padded inputs.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Bielik model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses the GGUF dequantization default.

        Returns:
            torch.nn.Module: The Bielik model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

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
        """Load and return sample inputs for the Bielik model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        self.seq_len = inputs["input_ids"].shape[1]

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the Bielik model variant.

        The config is embedded in the GGUF file rather than a separate
        config.json, so it is read by dequantizing the GGUF header via
        AutoConfig.

        Returns:
            The configuration object for the Bielik model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config

    def decode_output(self, outputs):
        """Decode model outputs into the next-token string.

        Args:
            outputs: Model output from a forward pass.

        Returns:
            str: Decoded next token text.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
