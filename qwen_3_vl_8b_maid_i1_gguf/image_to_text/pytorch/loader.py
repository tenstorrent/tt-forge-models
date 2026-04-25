# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 8B Maid i1 GGUF model loader implementation for image to text.
"""

import torch.nn as nn
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
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

    Qwen3VLVisionConfig.out_hidden_size defaults to 3584, but for the 8B
    model it must equal the LM hidden_size (4096). The GGUF file stores only
    the qwen3 (text) architecture so this default is never overridden during
    GGUF loading. We fix the merger layers directly after from_pretrained.
    """
    lm_hidden = model.language_model.config.hidden_size
    mergers = [model.visual.merger] + list(
        getattr(model.visual, "deepstack_merger_list", [])
    )
    for merger in mergers:
        fc2 = merger.linear_fc2
        if fc2.out_features != lm_hidden:
            new_fc2 = nn.Linear(fc2.in_features, lm_hidden, bias=fc2.bias is not None)
            new_fc2.to(device=fc2.weight.device, dtype=fc2.weight.dtype)
            merger.linear_fc2 = new_fc2
    model.config.vision_config.out_hidden_size = lm_hidden


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 8B Maid i1 GGUF model variants for image to text."""

    QWEN_3_VL_8B_MAID_I1_GGUF = "8b_maid_i1_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 8B Maid i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_8B_MAID_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-8B-maid-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_8B_MAID_I1_GGUF

    GGUF_FILE = "Qwen3-VL-8B-maid.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 8B Maid i1 GGUF",
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
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        )
        _fix_merger_out_hidden_size(model)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
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
