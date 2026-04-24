# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
amlm_hard_nhot model loader implementation for masked language modeling.
"""

from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available amlm_hard_nhot model variants for masked language modeling."""

    AMLM_HARD_NHOT = "leukas/amlm_hard_nhot"


class ModelLoader(ForgeModel):
    """amlm_hard_nhot model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.AMLM_HARD_NHOT: LLMModelConfig(
            pretrained_model_name="leukas/amlm_hard_nhot",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AMLM_HARD_NHOT

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "The capital of France is [MASK]."
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="amlm_hard_nhot",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self.model_name, **model_kwargs)

        # In transformers 5.x, models are always initialized on meta device. The custom
        # NGramEmbeds.ref_table is not a registered buffer so it stays as a meta tensor
        # and cannot be copied in forward(). Re-initialize it with real data after loading,
        # casting to match the model's parameter dtype.
        try:
            ngram_embeds = model.deberta.NGram_embeddings
            param_dtype = next(model.parameters()).dtype
            ngram_embeds.ref_table = ngram_embeds.prepare_vocab_table().to(param_dtype)
        except (AttributeError, StopIteration):
            pass

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        print("The predicted token for the [MASK] is:", predicted_token)

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
