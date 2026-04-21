# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pythia reward model loader implementation for sequence classification.
"""
from typing import Optional

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Pythia reward model variants."""

    PYTHIA_1B_DEDUPED_TLDR = "1B-deduped-tldr"


class ModelLoader(ForgeModel):
    """Pythia reward model loader implementation."""

    _VARIANTS = {
        ModelVariant.PYTHIA_1B_DEDUPED_TLDR: ModelConfig(
            pretrained_model_name="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PYTHIA_1B_DEDUPED_TLDR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pythia-Reward",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        input_text = (
            "SUBREDDIT: r/AskReddit\n"
            "TITLE: What is the best way to learn a new language?\n"
            "POST: I've been trying to learn Spanish for a few months now but I'm not making "
            "much progress. Any advice on how to improve?\n"
            "TL;DR: Struggling to learn Spanish, looking for advice on how to improve."
        )

        inputs = self.tokenizer(
            input_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        reward_score = logits.item()
        print(f"Reward score: {reward_score:.4f}")
