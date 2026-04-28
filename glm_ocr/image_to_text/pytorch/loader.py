# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-OCR model loader implementation for image-to-text tasks.
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
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


def _patch_glm_ocr_for_tt_device():
    """Patch GLM-OCR methods that call .tolist() or iterate on device tensors.

    TT device does not support eager tensor reads — .tolist() or iterating over
    a TT tensor triggers a device sync that fails with Error code: 13. Move the
    small integer metadata tensors (grid_thw, input_ids) to CPU before those
    calls; move computed outputs (position_ids, rope_deltas) back to the
    original device so the language model path stays on TT device.
    """
    try:
        from transformers.models.glm_ocr import modeling_glm_ocr
    except ImportError:
        return

    orig_rot_pos = modeling_glm_ocr.GlmOcrVisionModel.rot_pos_emb
    orig_get_rope = modeling_glm_ocr.GlmOcrModel.get_rope_index
    orig_get_image = modeling_glm_ocr.GlmOcrModel.get_image_features
    orig_get_video = modeling_glm_ocr.GlmOcrModel.get_video_features

    def _patched_rot_pos(self, grid_thw):
        return orig_rot_pos(self, grid_thw.cpu())

    def _patched_get_rope(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        orig_device = input_ids.device if input_ids is not None else None
        position_ids, rope_deltas = orig_get_rope(
            self,
            input_ids=input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            **kwargs,
        )
        if orig_device is not None:
            position_ids = position_ids.to(orig_device)
            rope_deltas = rope_deltas.to(orig_device)
        return position_ids, rope_deltas

    def _patched_get_image(self, pixel_values, image_grid_thw=None, **kwargs):
        return orig_get_image(
            self,
            pixel_values,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            **kwargs,
        )

    def _patched_get_video(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return orig_get_video(
            self,
            pixel_values_videos,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            **kwargs,
        )

    modeling_glm_ocr.GlmOcrVisionModel.rot_pos_emb = _patched_rot_pos
    modeling_glm_ocr.GlmOcrModel.get_rope_index = _patched_get_rope
    modeling_glm_ocr.GlmOcrModel.get_image_features = _patched_get_image
    modeling_glm_ocr.GlmOcrModel.get_video_features = _patched_get_video


class ModelVariant(StrEnum):
    """Available GLM-OCR model variants for image-to-text tasks."""

    GLM_OCR = "glm_ocr"
    GLM_OCR_MLX_8BIT = "mlx_8bit"


class ModelLoader(ForgeModel):
    """GLM-OCR model loader implementation for image-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GLM_OCR: LLMModelConfig(
            pretrained_model_name="zai-org/GLM-OCR",
        ),
        ModelVariant.GLM_OCR_MLX_8BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/GLM-OCR-8bit",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GLM_OCR

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
        return ModelInfo(
            model="glm_ocr",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load Processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance
        """
        kwargs = {"use_fast": False}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GLM-OCR model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GLM-OCR model instance for image-to-text tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        _patch_glm_ocr_for_tt_device()

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GLM-OCR model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Text Recognition:"},
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
        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs
