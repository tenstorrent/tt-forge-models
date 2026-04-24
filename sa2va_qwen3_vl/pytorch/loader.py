# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sa2VA-Qwen3-VL model loader implementation for multimodal visual question answering.
"""

import json

import safetensors.torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
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
    """Available Sa2VA-Qwen3-VL model variants."""

    SA2VA_QWEN3_VL_4B = "4b"


class ModelLoader(ForgeModel):
    """Sa2VA-Qwen3-VL model loader for multimodal visual question answering built on Qwen3-VL."""

    _VARIANTS = {
        ModelVariant.SA2VA_QWEN3_VL_4B: ModelConfig(
            pretrained_model_name="ByteDance/Sa2VA-Qwen3-VL-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SA2VA_QWEN3_VL_4B

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Sa2VA-Qwen3-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        # Sa2VA's SAM2 component calls .item() during __init__, which is
        # incompatible with transformers 5.x meta-device initialization.
        # Manually instantiate the wrapper class on CPU, load sharded weights,
        # then return the inner Qwen3VLForConditionalGeneration model (which has
        # a standard forward method that matches our input format).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"],
            pretrained_model_name,
            trust_remote_code=True,
        )
        wrapper = model_class(config)

        index_file = hf_hub_download(
            pretrained_model_name, "model.safetensors.index.json"
        )
        with open(index_file) as f:
            index = json.load(f)

        shard_files = sorted(set(index["weight_map"].values()))
        state_dict = {}
        for shard_file in shard_files:
            shard_path = hf_hub_download(pretrained_model_name, shard_file)
            state_dict.update(safetensors.torch.load_file(shard_path))

        wrapper.load_state_dict(state_dict, strict=False)

        # The outer Sa2VAChatModelQwen has no standard forward(); use the inner
        # Qwen3VLForConditionalGeneration whose forward accepts the processor outputs.
        model = wrapper.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image},
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

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
