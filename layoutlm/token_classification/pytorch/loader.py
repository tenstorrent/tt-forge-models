# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LayoutLM token classification model loader implementation (PyTorch).
"""

import torch
from transformers import AutoTokenizer, LayoutLMForTokenClassification
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available LayoutLM token classification model variants."""

    ALLENAI_IVILA_ROW_LAYOUTLM_FINETUNED_S2VL_V2 = (
        "allenai/ivila-row-layoutlm-finetuned-s2vl-v2"
    )


class ModelLoader(ForgeModel):
    """LayoutLM token classification model loader implementation."""

    _VARIANTS = {
        ModelVariant.ALLENAI_IVILA_ROW_LAYOUTLM_FINETUNED_S2VL_V2: ModelConfig(
            pretrained_model_name="allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALLENAI_IVILA_ROW_LAYOUTLM_FINETUNED_S2VL_V2

    # Sample document words and their bounding boxes (normalized 0-1000)
    words = ["Invoice", "Number:", "12345", "Date:", "2024-01-15"]
    boxes = [
        [100, 50, 200, 80],
        [210, 50, 330, 80],
        [340, 50, 420, 80],
        [100, 100, 180, 130],
        [190, 100, 340, 130],
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LayoutLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LayoutLMForTokenClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = 512

        # LayoutLM v1 tokenizer does not accept boxes; tokenize words and
        # manually align bounding boxes to the resulting word-piece tokens.
        encoding = self.tokenizer(
            self.words,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        # Build bbox tensor aligned with tokenized output.
        # [CLS] and [SEP] get [0,0,0,0]; each word-piece inherits its word's box.
        word_ids = encoding.word_ids(batch_index=0)
        bbox = []
        for wid in word_ids:
            if wid is None:
                bbox.append([0, 0, 0, 0])
            else:
                bbox.append(self.boxes[wid])
        bbox = torch.tensor([bbox], dtype=torch.long)

        inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": bbox,
            "token_type_ids": encoding["token_type_ids"],
        }

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for token classification."""
        inputs = self.load_inputs()
        logits = co_out[0]
        predicted_token_class_ids = logits.argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        print(f"Words: {self.words}")
        print(f"Predicted classes: {predicted_tokens_classes}")
