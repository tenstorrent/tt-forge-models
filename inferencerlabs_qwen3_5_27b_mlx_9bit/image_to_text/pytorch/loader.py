# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
inferencerlabs/Qwen3.5-27B-MLX-9bit model loader implementation for image to text.
"""

import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from typing import Optional

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


def _fix_merger_out_hidden_size(model):
    """Fix vision merger linear_fc2 output dim to match LM hidden_size.

    Qwen3_5VisionConfig.out_hidden_size defaults to 3584, but for this
    27B model it must equal the LM hidden_size. The MLX repo config may
    not set this correctly, so we rebuild linear_fc2 post-load.
    """
    lm_hidden = model.config.text_config.hidden_size
    visual = model.model.visual
    mergers = [visual.merger] + list(getattr(visual, "deepstack_merger_list", []))
    for merger in mergers:
        fc2 = merger.linear_fc2
        if fc2.out_features != lm_hidden:
            new_fc2 = nn.Linear(fc2.in_features, lm_hidden, bias=fc2.bias is not None)
            new_fc2.to(device=fc2.weight.device, dtype=fc2.weight.dtype)
            merger.linear_fc2 = new_fc2
    model.config.vision_config.out_hidden_size = lm_hidden


class ModelVariant(StrEnum):
    """Available inferencerlabs Qwen3.5-27B-MLX-9bit model variants for image to text."""

    QWEN_3_5_27B_MLX_9BIT = "27B_MLX_9bit"


class ModelLoader(ForgeModel):
    """inferencerlabs Qwen3.5-27B-MLX-9bit model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B_MLX_9BIT: LLMModelConfig(
            pretrained_model_name="inferencerlabs/Qwen3.5-27B-MLX-9bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B_MLX_9BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="inferencerlabs Qwen3.5-27B-MLX-9bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        # MLX repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-27B")

        # MLX quantization config lacks quant_method; remove it to load as bf16
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        model_kwargs["config"] = config

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        _fix_merger_out_hidden_size(model)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
