# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PhoRanker model loader implementation for passage ranking.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available PhoRanker model variants for passage ranking."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """PhoRanker model loader implementation for passage ranking."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="itdainb/PhoRanker",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Sample query-passage pairs for testing (Vietnamese word-segmented text).
    # PhoRanker expects input that has already been word-segmented (VnCoreNLP style).
    sample_pairs = [
        (
            "Trường UIT là gì ?",
            "Trường Đại_học Công_nghệ Thông_tin có tên tiếng Anh là University of Information_Technology ( viết tắt là UIT ) là thành_viên của Đại_học Quốc_Gia TP. HCM .",
        ),
        (
            "Trường UIT là gì ?",
            "Trường Đại_học Kinh_tế – Luật ( tiếng Anh : University of Economics and Law – UEL ) là trường đại_học đào_tạo và nghiên_cứu khối ngành kinh_tế , kinh_doanh và luật hàng_đầu Việt_Nam .",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PhoRanker",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        queries = [pair[0] for pair in self.sample_pairs]
        passages = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
