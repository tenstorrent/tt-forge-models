# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BeetleLM model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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


class ModelVariant(StrEnum):
    """Available BeetleLM model variants for causal language modeling."""

    BEETLELM_DEU_L1_ENG_L2_BALANCED = "beetlelm_deu_L1_eng_L2_balanced"
    BEETLELM_NLD_BUL_BALANCED = "beetlelm_nld-bul_balanced"


class ModelLoader(ForgeModel):
    """BeetleLM model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BEETLELM_DEU_L1_ENG_L2_BALANCED: LLMModelConfig(
            pretrained_model_name="BeetleLM/beetlelm_deu_L1-eng_L2_balanced",
            max_length=128,
        ),
        ModelVariant.BEETLELM_NLD_BUL_BALANCED: LLMModelConfig(
            pretrained_model_name="BeetleLM/beetlelm_nld-bul_balanced",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BEETLELM_DEU_L1_ENG_L2_BALANCED

    # Shared configuration parameters
    sample_text = "The quick brown fox jumps over the lazy dog."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

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
            model="BeetleLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # BeetleLM repos don't include tokenizer files; use open_llama's matching vocabulary
    _TOKENIZER_SOURCE = "openlm-research/open_llama_3b_v2"

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_SOURCE)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BeetleLM model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BeetleLM model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # BeetleLM repos contain no weight files; instantiate from config with random weights
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            config.n_layers = self.num_layers

        # from_config with trust_remote_code needs _name_or_path to locate the custom module
        config._name_or_path = pretrained_model_name
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BeetleLM model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
