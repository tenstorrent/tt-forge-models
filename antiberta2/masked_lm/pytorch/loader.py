# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AntiBERTa2 model loader implementation for masked language modeling.

AntiBERTa2 is an antibody-specific RoFormer-based language model pre-trained
using masked language modelling on antibody amino acid sequences.

Reference: https://huggingface.co/alchemab/antiberta2
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available AntiBERTa2 model variants for masked language modeling."""

    ANTIBERTA2 = "alchemab/antiberta2"


class ModelLoader(ForgeModel):
    """AntiBERTa2 model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.ANTIBERTA2: LLMModelConfig(
            pretrained_model_name="alchemab/antiberta2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANTIBERTA2

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        # Heavy chain antibody sequence with masked residue, per model card.
        self.sample_text = (
            "Q V Q L V Q S G A E V K K P G A S V K V S C K A S G Y T F T "
            "S Y G I S W V R Q A P G Q G L E W M G W I S A Y N G N T N Y "
            "A Q K L Q G R V T M T T D T S T S T A Y M E L R S L R S D D "
            "T A V Y Y C A R [MASK] G G F D Y W G Q G T L V T V S S"
        )
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="AntiBERTa2",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
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
        """Decode the model output for masked language modeling."""
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
            self._variant_config.pretrained_model_name
        )
        return self.config
