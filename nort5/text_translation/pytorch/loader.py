# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
NorT5 model loader implementation for English-Norwegian text translation.
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
    """Available NorT5 text translation model variants."""

    BASE_EN_NO = "Base_En_No"


class ModelLoader(ForgeModel):
    """NorT5 model loader implementation for English-Norwegian text translation."""

    _VARIANTS = {
        ModelVariant.BASE_EN_NO: LLMModelConfig(
            pretrained_model_name="ltg/nort5-base-en-no-translation",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_EN_NO

    sample_text = "How are you feeling right now? Better?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="NorT5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NorT5 model instance."""
        from transformers import AutoModelForSeq2SeqLM
        from transformers.modeling_utils import PreTrainedModel

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"trust_remote_code": True, "return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # NorT5's custom __init__ doesn't call post_init(), which is required in
        # transformers v5.x to set all_tied_weights_keys. Patch the method to initialize
        # the attribute if missing before it's accessed during model loading.
        _orig_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers

        def _patched_adjust(self, *args, **kwargs):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
                    all_submodels=False
                )
            return _orig_adjust(self, *args, **kwargs)

        PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name, **model_kwargs
            )
        finally:
            PreTrainedModel._adjust_tied_keys_with_tied_pointers = _orig_adjust

        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the NorT5 model."""
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # NorT5 uses control tokens [CLS] >>target_lang<< >>source_lang<< ... [SEP]
        # to indicate translation direction. Here we translate English -> Bokmal.
        cls_index = self._tokenizer.convert_tokens_to_ids("[CLS]")
        sep_index = self._tokenizer.convert_tokens_to_ids("[SEP]")
        eng_index = self._tokenizer.convert_tokens_to_ids(">>eng<<")
        nob_index = self._tokenizer.convert_tokens_to_ids(">>nob<<")

        source_subwords = self._tokenizer(self.sample_text).input_ids
        source_subwords = (
            [cls_index, nob_index, eng_index] + source_subwords + [sep_index]
        )
        input_ids = torch.tensor([source_subwords], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Seq2seq models need decoder_input_ids for the forward pass.
        # NorT5Config doesn't set decoder_start_token_id; fall back to bos_token_id.
        decoder_start_token_id = (
            getattr(self._model.config, "decoder_start_token_id", None)
            or self._model.config.bos_token_id
        )
        decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
