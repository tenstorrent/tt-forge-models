# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Perception-LM model loader implementation for image-text-to-text generation.
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageTextToText, AutoProcessor

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Perception-LM model variants."""

    PERCEPTION_LM_1B = "facebook/Perception-LM-1B"


class ModelLoader(ForgeModel):
    """Perception-LM model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.PERCEPTION_LM_1B: ModelConfig(
            pretrained_model_name=str(ModelVariant.PERCEPTION_LM_1B),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PERCEPTION_LM_1B

    sample_text = "Describe the bar plot in the image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Perception-LM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, use_fast=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        test_image_file = hf_hub_download(
            repo_id="shumingh/perception_lm_test_images",
            filename="14496_0.PNG",
            repo_type="dataset",
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": test_image_file},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.processor is None:
            self._load_processor()

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.processor.decode(token_ids[0], skip_special_tokens=True)
