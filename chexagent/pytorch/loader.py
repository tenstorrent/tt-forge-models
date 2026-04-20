# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CheXagent model loader implementation for chest X-ray vision-language tasks.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available CheXagent model variants."""

    CHEXAGENT_2_3B = "StanfordAIMI/CheXagent-2-3b"


class ModelLoader(ForgeModel):
    """CheXagent model loader implementation for chest X-ray vision-language tasks."""

    _VARIANTS = {
        ModelVariant.CHEXAGENT_2_3B: ModelConfig(
            pretrained_model_name=str(ModelVariant.CHEXAGENT_2_3B),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHEXAGENT_2_3B

    sample_image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/4/4e/Lung_x-ray.jpg"
    )
    sample_text = "Describe the findings in this chest X-ray."
    sample_system_prompt = "You are a helpful assistant."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CheXagent",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        query = self.tokenizer.from_list_format(
            [
                {"image": self.sample_image_url},
                {"text": self.sample_text},
            ]
        )

        conv = [
            {"from": "system", "value": self.sample_system_prompt},
            {"from": "human", "value": query},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )

        return {"input_ids": input_ids}
