# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite model loader implementation for causal language modeling.
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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Granite model variants for causal LM."""

    # IBM Granite 4.0 variants
    GRANITE_4_0_MICRO = "granite_4_0_micro"


class ModelLoader(ForgeModel):
    """Granite model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GRANITE_4_0_MICRO: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-micro",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_MICRO

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
            model="Granite",
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

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        # Ensure a pad token is available for batched/padded inputs.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Granite model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its default dtype.

        Returns:
            torch.nn.Module: The Granite model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        # Granite applies several scalar multipliers (embedding/residual/attention
        # /logits) that amplify low-precision rounding error. Running in float32 by
        # default keeps the device output within PCC tolerance of the CPU reference;
        # callers may still request bfloat16 explicitly via dtype_override.
        dtype = dtype_override if dtype_override is not None else torch.float32

        model_kwargs = {"torch_dtype": dtype}
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Disable KV caching for single-pass inference. The granitemoehybrid
        # architecture builds a hybrid (attention + mamba) cache whose mamba-mask
        # path raises when the config has no linear-attention layers (as in the
        # all-attention micro variant). A plain forward needs no cache anyway.
        model.config.use_cache = False

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Granite model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the default dtype of input tensors.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Tokenize at natural length (no padding). Padded/masked positions produce
        # unconstrained logits that diverge between CPU and device and would drag
        # down the overall PCC, so every position fed to the model is a real token.
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        self.seq_len = inputs["input_ids"].shape[-1]
        return inputs

    def load_config(self):
        """Load and return the configuration for the Granite model variant.

        Returns:
            The configuration object for the Granite model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
