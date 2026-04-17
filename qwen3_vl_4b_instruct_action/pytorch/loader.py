# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-VL-4B-Instruct-action model loader implementation for image to text.
"""

import importlib


def _patch_qwen3_vl_source():
    """Patch the installed transformers source to guard repeat(1, 1).

    The TT XLA backend's repeat implementation uses concatenate, which fails
    when repeat factors are all 1 (no-op produces zero concatenation args).
    Must be called BEFORE importing any Qwen3VL classes.
    """
    import transformers.models.qwen3_vl.modeling_qwen3_vl as mod

    import inspect

    src_file = inspect.getfile(mod)
    with open(src_file, "r") as f:
        src = f.read()

    old = "            pos_embed = pos_embed.repeat(t, 1)\n"
    new = "            if t > 1:\n                pos_embed = pos_embed.repeat(t, 1)\n"
    if old in src and new not in src:
        src = src.replace(old, new)
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


class Qwen3VLWrapper(torch.nn.Module):
    """Wrapper that substitutes image_grid_thw with cached CPU values.

    In compile-only mode, integer tensors on XLA are zero-initialized.
    The vision encoder uses grid_thw values for data-dependent control flow
    (loop bounds, tensor shapes), so it cannot function with zeros.
    This wrapper ensures the correct values are always used.
    """

    def __init__(self, model, grid_thw_cpu):
        super().__init__()
        self.model = model
        self.register_buffer("_grid_thw_cpu", grid_thw_cpu.clone(), persistent=False)

    def forward(self, **kwargs):
        if "image_grid_thw" in kwargs and kwargs["image_grid_thw"] is not None:
            kwargs["image_grid_thw"] = self._grid_thw_cpu.to(
                dtype=kwargs["image_grid_thw"].dtype
            )
        return self.model(**kwargs)


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

        sample_inputs = self.processor.apply_chat_template(
            SAMPLE_MESSAGES,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        grid_thw_cpu = sample_inputs["image_grid_thw"]

        return Qwen3VLWrapper(model, grid_thw_cpu)

    def load_inputs(self, dtype_override=None, batch_size=1):
        inputs = self.processor.apply_chat_template(
            SAMPLE_MESSAGES,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
