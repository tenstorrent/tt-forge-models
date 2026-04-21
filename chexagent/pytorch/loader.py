# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CheXagent model loader implementation for chest X-ray vision-language tasks.
"""

import transformers
import transformers.utils

# CheXagent's custom modules require transformers==4.40.0 API compatibility patches:
# - is_tf_available was removed in transformers 5.x
# - modeling_visual.py has a hard version assertion
if not hasattr(transformers.utils, "is_tf_available"):
    transformers.utils.is_tf_available = lambda: False
transformers.__version__ = "4.40.0"

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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

    CHEXAGENT_2_3B = "chexagent_2_3b"


class ModelLoader(ForgeModel):
    """CheXagent model loader implementation for chest X-ray vision-language tasks."""

    _VARIANTS = {
        ModelVariant.CHEXAGENT_2_3B: ModelConfig(
            pretrained_model_name="StanfordAIMI/CheXagent-2-3b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHEXAGENT_2_3B

    sample_image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
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

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        # pad_token_id was removed from PretrainedConfig defaults in transformers 5.x
        # but CheXagent's modeling code accesses it unconditionally
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
        # transformers 5.x changed rope_scaling format: old code expects {"type": ..., "factor": ...}
        # but new configs have {"rope_type": "default", ...} — treat "default" as no scaling
        if (
            isinstance(config.rope_scaling, dict)
            and config.rope_scaling.get("rope_type") == "default"
        ):
            config.rope_scaling = None

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
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

        result = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers 5.x returns a BatchEncoding dict; older versions returned a tensor
        input_ids = result["input_ids"] if isinstance(result, dict) else result

        return {"input_ids": input_ids}
