# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EuroBERT model loader implementation for token classification (NER).
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

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


def _compute_default_rope_parameters(
    config=None, device=None, seq_len=None, layer_type=None
):
    rope_theta = 10000.0
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        rope_theta = config.rope_scaling.get("rope_theta", rope_theta)
    elif hasattr(config, "rope_theta") and config.rope_theta is not None:
        rope_theta = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0) or 1.0
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


class ModelVariant(StrEnum):
    """Available EuroBERT token classification model variants."""

    OPENMED_NER_PHARMADETECT_EUROMED_212M = "OpenMed-NER-PharmaDetect-EuroMed-212M"


class ModelLoader(ForgeModel):
    """EuroBERT model loader implementation for token classification (NER)."""

    _VARIANTS = {
        ModelVariant.OPENMED_NER_PHARMADETECT_EUROMED_212M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-PharmaDetect-EuroMed-212M",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_NER_PHARMADETECT_EUROMED_212M

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None
        self.sample_text = (
            "Administration of metformin reduced glucose levels significantly."
        )

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EuroBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, trust_remote_code=True, **model_kwargs
        )
        self.model.eval()
        return self.model

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
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        print(f"Context: {self.sample_text}")
        print(f"Predicted Labels: {predicted_tokens_classes}")
