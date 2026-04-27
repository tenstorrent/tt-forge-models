# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlinVLM model loader implementation for image to text tasks.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
)
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

# ---------------------------------------------------------------------------
# Monkey-patches for Qwen3VL on TT hardware
#
# Two root causes prevent compilation:
#
#  1. Qwen3VLVisionModel uses nn.Conv3d (patch embedding) and sequences of up
#     to 11,008 tokens that overflow TT L1 (max 1.5 MB per core).  Fix: run
#     the visual encoder eagerly on CPU via torch.compiler.disable.
#
#  2. Several methods call .tolist() on tensors that may land on TT device
#     (image_grid_thw, input_ids).  Fix: move them to CPU before calling
#     the original methods; get_rope_index is also disabled so its control-
#     flow runs eagerly on CPU.
# ---------------------------------------------------------------------------

_orig_visual_forward = Qwen3VLVisionModel.forward


@torch.compiler.disable(recursive=True)
def _patched_visual_forward(self, hidden_states, grid_thw, **kwargs):
    """Run visual encoder eagerly on CPU to avoid TT L1 overflow and Conv3d."""
    param = next(self.parameters(), None)
    if param is not None and param.device.type != "cpu":
        self.cpu()
    if hidden_states.device.type != "cpu":
        hidden_states = hidden_states.cpu()
    if grid_thw.device.type != "cpu":
        grid_thw = grid_thw.cpu()
    return _orig_visual_forward(self, hidden_states, grid_thw, **kwargs)


Qwen3VLVisionModel.forward = _patched_visual_forward

_orig_get_image_features = Qwen3VLModel.get_image_features


def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
    """Move image_grid_thw to CPU for the split_sizes .tolist() call."""
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.cpu()
    return _orig_get_image_features(self, pixel_values, image_grid_thw, **kwargs)


Qwen3VLModel.get_image_features = _patched_get_image_features

_orig_get_rope_index = Qwen3VLModel.get_rope_index


@torch.compiler.disable(recursive=True)
def _patched_get_rope_index(
    self,
    input_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    **kwargs,
):
    """Run rope-index computation eagerly on CPU (uses .tolist() for control flow)."""
    if input_ids is not None and input_ids.device.type != "cpu":
        input_ids = input_ids.cpu()
    if image_grid_thw is not None and image_grid_thw.device.type != "cpu":
        image_grid_thw = image_grid_thw.cpu()
    if video_grid_thw is not None and video_grid_thw.device.type != "cpu":
        video_grid_thw = video_grid_thw.cpu()
    if attention_mask is not None and attention_mask.device.type != "cpu":
        attention_mask = attention_mask.cpu()
    return _orig_get_rope_index(
        self, input_ids, image_grid_thw, video_grid_thw, attention_mask, **kwargs
    )


Qwen3VLModel.get_rope_index = _patched_get_rope_index


class ModelVariant(StrEnum):
    """Available AlinVLM model variants for image to text."""

    ALINVLM_V1_3 = "v1_3"


class ModelLoader(ForgeModel):
    """AlinVLM model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.ALINVLM_V1_3: LLMModelConfig(
            pretrained_model_name="huiwon/alinvlm_v1_3",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ALINVLM_V1_3

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AlinVLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AlinVLM model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The AlinVLM model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"dtype": "auto", "device_map": "auto"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # use_fast=False: Qwen2VLImageProcessorFast ignores max_pixels; the
        # slow processor respects it and keeps the patch count manageable.
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name, use_fast=False
        )
        # Cap image resolution so the visual encoder's attention does not
        # overflow TT hardware L1 (1.5 MB per core).
        # 512×512 → ~988 patches vs 11,008 at native resolution.
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 512 * 512

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AlinVLM model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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
