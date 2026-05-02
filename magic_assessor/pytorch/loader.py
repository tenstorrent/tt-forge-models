# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MagicAssessor model loader implementation for vision-language tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional


def _patch_qwen2_5_vl():
    # XLA repeat_interleave tile-aligns integer values (e.g. 2204 → 2208) which breaks:
    #   (a) split_sizes in get_image_features: prod(-1).tolist() returns wrong values
    #   (b) cu_seqlens in vision transformer: grid_thw-based tolist() returns wrong values
    # Fix (a): move image_grid_thw to CPU at get_image_features entry.
    # Fix (b): move grid_thw to CPU at vision forward entry.
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as _qvl

    _orig_get_image_features = _qvl.Qwen2_5_VLForConditionalGeneration.get_image_features

    def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.cpu()
        return _orig_get_image_features(self, pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    _qvl.Qwen2_5_VLForConditionalGeneration.get_image_features = _patched_get_image_features

    _orig_vis_fwd = _qvl.Qwen2_5_VisionTransformerPretrainedModel.forward

    def _patched_vis_fwd(self, hidden_states, grid_thw, **kwargs):
        return _orig_vis_fwd(self, hidden_states, grid_thw.cpu(), **kwargs)

    _qvl.Qwen2_5_VisionTransformerPretrainedModel.forward = _patched_vis_fwd


_patch_qwen2_5_vl()

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
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available MagicAssessor model variants for vision-language tasks."""

    MAGIC_ASSESSOR_7B = "7B"


class ModelLoader(ForgeModel):
    """MagicAssessor model loader implementation for vision-language tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MAGIC_ASSESSOR_7B: LLMModelConfig(
            pretrained_model_name="wj-inf/MagicAssessor-7B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MAGIC_ASSESSOR_7B

    # Shared configuration parameters
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

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MagicAssessor",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
