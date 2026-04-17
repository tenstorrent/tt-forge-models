# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-VL-4B-Instruct-action model loader implementation for image to text.
"""

import importlib


def _patch_qwen3_vl_source():
    """Patch the installed transformers Qwen3VL source for TT XLA compatibility.

    Two fixes:
    1. Guard repeat(t, 1) with t > 1 to avoid empty concatenation.
    2. Cast grid_thw to long at vision model entry. This allows load_inputs
       to pass grid_thw as float so values survive XLA device transfer
       (integer tensors get zero-initialized in compile-only mode).
    """
    import transformers.models.qwen3_vl.modeling_qwen3_vl as mod
    import inspect

    src_file = inspect.getfile(mod)
    with open(src_file, "r") as f:
        src = f.read()

    modified = False

    # Fix 1: Guard repeat(1, 1) which fails on TT XLA backend
    old_repeat = "            pos_embed = pos_embed.repeat(t, 1)\n"
    new_repeat = (
        "            if t > 1:\n                pos_embed = pos_embed.repeat(t, 1)\n"
    )
    if old_repeat in src and new_repeat not in src:
        src = src.replace(old_repeat, new_repeat)
        modified = True

    # Fix 2: Cast image_grid_thw to long at conditional generation forward entry.
    # load_inputs passes grid_thw as float so values survive XLA device transfer.
    old_forward_entry = '        """\n\n        outputs = self.model(\n'
    new_forward_entry = (
        '        """\n'
        "        if image_grid_thw is not None:\n"
        "            image_grid_thw = image_grid_thw.long()\n"
        "\n        outputs = self.model(\n"
    )
    if old_forward_entry in src and "image_grid_thw = image_grid_thw.long()" not in src:
        src = src.replace(old_forward_entry, new_forward_entry, 1)
        modified = True

    if modified:
        with open(src_file, "w") as f:
            f.write(src)
        importlib.reload(mod)


_patch_qwen3_vl_source()

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from transformers import AutoProcessor
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


SAMPLE_MESSAGES = [
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


class ModelVariant(StrEnum):
    """Available Qwen3-VL-4B-Instruct-action model variants."""

    QWEN3_VL_4B_INSTRUCT_ACTION = "4b_instruct_action"


class ModelLoader(ForgeModel):
    """Qwen3-VL-4B-Instruct-action model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_4B_INSTRUCT_ACTION: LLMModelConfig(
            pretrained_model_name="229nagibator229/Qwen3-VL-4B-Instruct-action",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_4B_INSTRUCT_ACTION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="qwen3_vl_4b_instruct_action",
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

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, dtype="auto", device_map="auto", **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        inputs = self.processor.apply_chat_template(
            SAMPLE_MESSAGES,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Convert integer grid_thw to float so values survive XLA device transfer.
        # In compile-only mode, integer tensors are zero-initialized on XLA.
        # The patched vision model forward converts back to long.
        inputs["image_grid_thw"] = inputs["image_grid_thw"].float()
        return inputs
