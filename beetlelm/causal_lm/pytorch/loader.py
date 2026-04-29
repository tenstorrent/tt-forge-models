# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BeetleLM model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
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

    # BeetleLM model repos don't ship pretrained weights or tokenizers.
    # Tokenizers live in separate repos named bpe_babylm-<l1>-babylm-<l2>.
    _TOKENIZER_REPOS = {
        "BeetleLM/beetlelm_deu_L1-eng_L2_balanced": "BeetleLM/bpe_babylm-eng-babylm-deu",
        "BeetleLM/beetlelm_nld-bul_balanced": "BeetleLM/bpe_babylm-nld-babylm-bul",
    }

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

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer_repo = self._TOKENIZER_REPOS.get(pretrained_model_name, pretrained_model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repo,
            trust_remote_code=True,
        )

        # The BeetleLM tokenizer has two sets of special tokens: lowercase
        # (<PAD> ID=1) and uppercase ([PAD] ID=32002). The uppercase ones
        # exceed the model's vocab_size=32000. Use <PAD> for padding.
        if (
            self.tokenizer.pad_token_id is not None
            and self.tokenizer.pad_token_id >= 32000
            and "<PAD>" in self.tokenizer.get_vocab()
        ):
            self.tokenizer.pad_token = "<PAD>"

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

        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)

        if self.num_layers is not None:
            config.n_layers = self.num_layers

        # BeetleLM HuggingFace repos contain only config + architecture code,
        # no pretrained weights. Load the model class directly and initialize
        # with random weights so the compiler architecture test can proceed.
        model_class = get_class_from_dynamic_module(
            "pico_decoder.PicoDecoderHF", pretrained_model_name
        )
        model = model_class(config)

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
