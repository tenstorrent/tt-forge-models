# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InfiniteVL model loader implementation for image-text-to-text tasks.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from typing import Optional


def _default_rope_init(config, device=None):
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

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


class ModelVariant(StrEnum):
    """Available InfiniteVL model variants for image-text-to-text tasks."""

    INFINITEVL = "infinitevl"


class ModelLoader(ForgeModel):
    """InfiniteVL model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.INFINITEVL: LLMModelConfig(
            pretrained_model_name="hustvl/InfiniteVL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INFINITEVL

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="InfiniteVL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        # transformers 5.x removed pad_token_id default from PretrainedConfig;
        # patch sub-config so model __init__ doesn't raise AttributeError
        if hasattr(config, "text_config") and not hasattr(
            config.text_config, "pad_token_id"
        ):
            config.text_config.pad_token_id = None

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
