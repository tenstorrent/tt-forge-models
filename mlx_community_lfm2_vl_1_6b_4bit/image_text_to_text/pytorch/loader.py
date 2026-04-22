# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/LFM2-VL-1.6B-4bit model loader implementation for image-text-to-text tasks.
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from transformers.image_utils import load_image
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


class ModelVariant(StrEnum):
    """Available mlx-community LFM2-VL-1.6B-4bit model variants for image-text-to-text tasks."""

    LFM2_VL_1_6B_4BIT = "1_6B_4bit"


class ModelLoader(ForgeModel):
    """mlx-community LFM2-VL-1.6B-4bit model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_VL_1_6B_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/LFM2-VL-1.6B-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_VL_1_6B_4BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mlx-community LFM2-VL-1.6B-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        # use_fast=False avoids a Siglip2 fast-processor bug where tiled images
        # reach normalize with shape [N_tiles, 3, H, W] and cause a channel
        # mismatch (512 vs 3) when broadcasting the RGB mean.
        kwargs = {"trust_remote_code": True, "use_fast": False}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    @staticmethod
    def _patch_tied_weights_keys_compat():
        # transformers 5.x requires _tied_weights_keys to be a dict, not a list.
        # Models with trust_remote_code may still use the old list format.
        # Patch get_expanded_tied_weights_keys to convert list -> None before dispatch.
        from transformers.modeling_utils import PreTrainedModel

        if getattr(
            PreTrainedModel.get_expanded_tied_weights_keys, "_compat_patched", False
        ):
            return

        original = PreTrainedModel.get_expanded_tied_weights_keys

        def _compat(self, all_submodels=False):
            if not all_submodels and isinstance(
                getattr(self, "_tied_weights_keys", None), list
            ):
                self._tied_weights_keys = None
            return original(self, all_submodels=all_submodels)

        _compat._compat_patched = True
        PreTrainedModel.get_expanded_tied_weights_keys = _compat

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        self._patch_tied_weights_keys_compat()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
