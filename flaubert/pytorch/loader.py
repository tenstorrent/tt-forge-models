# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FlauBERT masked language model loader.
FlauBERT is a French language model pretrained on a large and heterogeneous French corpus.
"""
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from typing import Optional


class _NoCacheCompat:
    """Workaround for transformers 5.x FlauBERT pre_norm=True branch.

    The pre_norm branch calls cache[i] (subscript) but transformers 5.x
    now initializes cache as EncoderDecoderCache which is not subscriptable.
    This shim is not None (bypassing EncoderDecoderCache creation), returns 0
    from get_seq_length() so no tokens are skipped, and returns None on
    subscript so MultiHeadAttention skips caching entirely.
    """

    def get_seq_length(self):
        return 0

    def __getitem__(self, i):
        return None

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available FlauBERT model variants."""

    FLAUBERT_BASE_UNCASED = "Base_Uncased"
    FLAUBERT_SMALL_CASED = "Small_Cased"


class ModelLoader(ForgeModel):
    """FlauBERT masked language model loader."""

    _VARIANTS = {
        ModelVariant.FLAUBERT_BASE_UNCASED: ModelConfig(
            pretrained_model_name="flaubert/flaubert_base_uncased",
        ),
        ModelVariant.FLAUBERT_SMALL_CASED: ModelConfig(
            pretrained_model_name="flaubert/flaubert_small_cased",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLAUBERT_BASE_UNCASED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FlauBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        test_input = "Paris est la <special1> de la France."

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Transformers 5.x FlauBERT pre_norm=True branch calls cache[i] but
        # now initializes cache as EncoderDecoderCache (not subscriptable). Pass
        # a shim that satisfies all cache access patterns without actual caching.
        config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
        if getattr(config, "pre_norm", False):
            inputs["cache"] = _NoCacheCompat()

        return inputs

    def decode_output(self, outputs):
        if self.tokenizer is None:
            self._load_tokenizer()

        if isinstance(outputs, list):
            logits = outputs[0].logits if hasattr(outputs[0], "logits") else outputs[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        inputs = self.load_inputs()

        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]

        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

        output = self.tokenizer.decode(predicted_token_id)

        return f"Output: {output}"
