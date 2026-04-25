# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/LydiaTM-SKL-32B-i1-GGUF model loader implementation for image to text.
"""

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
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

# The GGUF repo ships a config.json for a smaller model (hidden=4096) but the GGUF
# weights are 32B (hidden=5120), causing shape mismatches.  Bypass GGUF loading and
# use from_config with the correct 32B config; limit layers so the
# randomly-initialized model fits in memory.
_DEFAULT_NUM_LAYERS = 4

# Base (non-GGUF) repo that has the correct 32B architecture config and processor
_BASE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


class ModelVariant(StrEnum):
    """Available mradermacher/LydiaTM-SKL-32B-i1-GGUF model variants for image to text."""

    LYDIATM_SKL_32B_I1_GGUF = "LydiaTM_SKL_32B_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/LydiaTM-SKL-32B-i1-GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.LYDIATM_SKL_32B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/LydiaTM-SKL-32B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LYDIATM_SKL_32B_I1_GGUF

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.processor = None
        self.num_layers = num_layers if num_layers is not None else _DEFAULT_NUM_LAYERS

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher LydiaTM-SKL-32B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(_BASE_MODEL)

        config = AutoConfig.from_pretrained(_BASE_MODEL)
        if hasattr(config, "text_config"):
            config.text_config.num_hidden_layers = self.num_layers
        else:
            config.num_hidden_layers = self.num_layers

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = AutoModelForImageTextToText.from_config(
            config, torch_dtype=dtype
        ).eval()
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
