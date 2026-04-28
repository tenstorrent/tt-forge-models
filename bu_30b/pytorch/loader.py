# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BU-30B model loader implementation for image to text.
"""

import torch
from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
)
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as _qvlmoe
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


def _apply_tolist_patches():
    """Patch qwen3_vl_moe methods that call .tolist() on device tensors.

    TT device tensors do not support eager Python-side readback (.tolist()).
    These methods use .tolist() only for control-flow on grid metadata or
    token ids, not for main model computation, so moving to CPU is safe.
    """
    _orig_fpe = _qvlmoe.Qwen3VLMoeVisionModel.fast_pos_embed_interpolate

    def _patched_fpe(self, grid_thw):
        return _orig_fpe(self, grid_thw.cpu())

    _qvlmoe.Qwen3VLMoeVisionModel.fast_pos_embed_interpolate = _patched_fpe

    _orig_rpe = _qvlmoe.Qwen3VLMoeVisionModel.rot_pos_emb

    def _patched_rpe(self, grid_thw):
        return _orig_rpe(self, grid_thw.cpu())

    _qvlmoe.Qwen3VLMoeVisionModel.rot_pos_emb = _patched_rpe

    _orig_gri = _qvlmoe.Qwen3VLMoeModel.get_rope_index

    def _patched_gri(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        orig_device = input_ids.device if input_ids is not None else torch.device("cpu")
        position_ids, rope_deltas = _orig_gri(
            self,
            input_ids=input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            **kwargs,
        )
        return position_ids.to(orig_device), rope_deltas.to(orig_device)

    _qvlmoe.Qwen3VLMoeModel.get_rope_index = _patched_gri

    def _patched_gif(self, pixel_values, image_grid_thw=None, **kwargs):
        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (
            image_grid_thw.cpu().prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    _qvlmoe.Qwen3VLMoeModel.get_image_features = _patched_gif


class ModelVariant(StrEnum):
    """Available BU-30B model variants for image to text."""

    BU_30B_A3B_PREVIEW = "30b_a3b_preview"


class ModelLoader(ForgeModel):
    """BU-30B model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BU_30B_A3B_PREVIEW: LLMModelConfig(
            pretrained_model_name="browser-use/bu-30b-a3b-preview",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BU_30B_A3B_PREVIEW

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
            model="bu_30b",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BU-30B model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BU-30B model instance for image to text.
        """
        _apply_tolist_patches()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.processor.image_processor.min_pixels = 56 * 56
        self.processor.image_processor.max_pixels = 13 * 28 * 1280

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BU-30B model.

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
