# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MBart50 model loader implementation for Ukrainian text verbalization.
"""

import torch
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
    """Available MBart50 model variants for text verbalization."""

    SKYPRO1111_UK = "skypro1111_uk"


class ModelLoader(ForgeModel):
    """MBart50 model loader implementation for Ukrainian text verbalization."""

    _VARIANTS = {
        ModelVariant.SKYPRO1111_UK: LLMModelConfig(
            pretrained_model_name="skypro1111/mbart-large-50-verbalization",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SKYPRO1111_UK

    sample_text = "<verbalization>:Цей додаток вийде 15.06.2025."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

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
            model="MBart50",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer instance
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._tokenizer.src_lang = "uk_XX"
        self._tokenizer.tgt_lang = "uk_XX"

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MBart50 verbalization model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import MBartForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MBartForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MBart50 verbalization model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        # Seq2seq models need decoder_input_ids for the forward pass.
        # The verbalization model targets Ukrainian, so start decoding from uk_XX.
        target_lang_id = self._tokenizer.lang_code_to_id["uk_XX"]
        inputs["decoder_input_ids"] = torch.tensor([[target_lang_id]])

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
