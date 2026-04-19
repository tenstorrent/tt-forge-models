# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FEEL-IT Italian Emotion model loader implementation for sequence classification.
"""

import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification
from transformers.models.camembert.tokenization_camembert import CamembertTokenizer
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
    """Available FEEL-IT Italian Emotion model variants for sequence classification."""

    FEEL_IT_ITALIAN_EMOTION = "feel_it_italian_emotion"


class ModelLoader(ForgeModel):
    """FEEL-IT Italian Emotion model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.FEEL_IT_ITALIAN_EMOTION: LLMModelConfig(
            pretrained_model_name="MilaNLProc/feel-it-italian-emotion",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FEEL_IT_ITALIAN_EMOTION

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.review = "Oggi sono proprio contento!"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="FEEL_IT_Italian_Emotion",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load FEEL-IT Italian Emotion model for sequence classification from Hugging Face."""

        spiece_path = hf_hub_download(self.model_name, "sentencepiece.bpe.model")
        sp = spm.SentencePieceProcessor()
        sp.Load(spiece_path)
        vocab = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]
        self.tokenizer = CamembertTokenizer(vocab=vocab)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for FEEL-IT Italian Emotion sequence classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for sequence classification."""
        predicted_value = co_out[0].argmax(-1).item()

        print(f"Predicted Emotion: {self.model.config.id2label[predicted_value]}")
