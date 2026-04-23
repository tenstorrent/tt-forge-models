# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moondream2 GGUF model loader implementation for image-text-to-text tasks.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from transformers import PhiConfig
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

_PHI2_GGUF_CONFIG_FIELDS = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_epsilon": "layer_norm_eps",
    "attention.rotary_pct": "partial_rotary_factor",
    "vocab_size": "vocab_size",
}


def _patch_phi2_support():
    """Register phi2 GGUF architecture for loading as transformers phi (phi-2) model."""
    if "phi2" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("phi2")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "phi2", _PHI2_GGUF_CONFIG_FIELDS
    )
    CONFIG_MAPPING._extra_content.setdefault("phi2", PhiConfig)
    if "gpt2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("phi2", GGUF_TO_FAST_CONVERTERS["gpt2"])


_patch_phi2_support()


class ModelVariant(StrEnum):
    """Available Moondream2 GGUF model variants for image-text-to-text tasks."""

    MOONDREAM2_F16 = "F16"


class ModelLoader(ForgeModel):
    """Moondream2 GGUF model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.MOONDREAM2_F16: LLMModelConfig(
            pretrained_model_name="moondream/moondream2-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOONDREAM2_F16

    GGUF_FILE = "moondream2-text-model-f16.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Moondream2 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        prompt = "What is shown in this image?"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        if self.config is not None:
            return self.config
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )
        return self.config
