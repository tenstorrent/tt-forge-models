# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sa2VA-Qwen3-VL model loader implementation for multimodal visual question answering.
"""

import torch
from transformers import AutoModel, AutoProcessor
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

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": False,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # SAM2/Hiera inside Sa2VA calls .item() on torch.linspace() during __init__,
        # which fails when transformers 5.x wraps initialization in a meta-device
        # context.  Strip that context so .item() works on real CPU tensors.
        #
        # Additionally, Sa2VAChatModelQwen never calls self.post_init(), so
        # all_tied_weights_keys is never set.  Transformers 5.x
        # _finalize_model_loading requires it.  Wrap _finalize_model_loading to
        # call post_init() on any model that skipped it.
        from transformers import PreTrainedModel

        _orig_get_init_context = PreTrainedModel.get_init_context.__func__
        _orig_finalize = PreTrainedModel._finalize_model_loading

        @classmethod
        def _no_meta_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            contexts = _orig_get_init_context(
                cls, dtype, is_quantized, _is_ds_init_called
            )
            return [
                c
                for c in contexts
                if not (isinstance(c, torch.device) and c.type == "meta")
            ]

        @staticmethod
        def _finalize_with_post_init(model, *args, **kwargs):
            if not hasattr(model, "all_tied_weights_keys"):
                model.post_init()
            return _orig_finalize(model, *args, **kwargs)

        PreTrainedModel.get_init_context = _no_meta_init_context
        PreTrainedModel._finalize_model_loading = _finalize_with_post_init
        try:
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        finally:
            PreTrainedModel.get_init_context = classmethod(_orig_get_init_context)
            PreTrainedModel._finalize_model_loading = staticmethod(_orig_finalize)
        model.eval()

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
