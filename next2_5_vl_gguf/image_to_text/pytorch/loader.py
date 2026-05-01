# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Next2.5 VL GGUF model loader implementation for image to text.
"""

from typing import Optional

from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, AutoConfig

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _patch_qwen3_5_for_tt_device():
    """Patch Qwen3.5 VL methods that call .tolist() on device tensors.

    TT device does not support eager tensor reads — any .tolist() on a TT
    tensor triggers a device sync that fails with Error code: 13. Patch the
    class methods directly so dynamo sees graph-breaks at the right points.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        return

    orig_fast_pos = modeling_qwen3_5.Qwen3_5VisionModel.fast_pos_embed_interpolate
    orig_rot_pos = modeling_qwen3_5.Qwen3_5VisionModel.rot_pos_emb
    orig_get_rope = modeling_qwen3_5.Qwen3_5Model.get_rope_index
    orig_get_image = modeling_qwen3_5.Qwen3_5Model.get_image_features

    def _patched_fast_pos(self, grid_thw):
        return orig_fast_pos(self, grid_thw.cpu())

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

    modeling_qwen3_5.Qwen3_5VisionModel.fast_pos_embed_interpolate = _patched_fast_pos
    modeling_qwen3_5.Qwen3_5VisionModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen3_5.Qwen3_5Model.get_rope_index = _patched_get_rope
    modeling_qwen3_5.Qwen3_5Model.get_image_features = _patched_get_image


class ModelVariant(StrEnum):
    """Available Next2.5 VL GGUF model variants."""

    NEXT2_5_I1_Q4_K_M = "i1_Q4_K_M"


class ModelLoader(ForgeModel):
    """Next2.5 VL GGUF model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.NEXT2_5_I1_Q4_K_M: ModelConfig(
            pretrained_model_name="thelamapi/next2.5-i1-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEXT2_5_I1_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.NEXT2_5_I1_Q4_K_M: "next2.5.i1-Q4_K_M.gguf",
    }

    _PROCESSOR_NAME = "thelamapi/next2.5"

    # Pixel limits to keep patch count within hardware budget
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Next2.5 VL GGUF model loader."""
        super().__init__(variant)
        self.processor = None
        self.config = None

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Next2.5 VL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self._PROCESSOR_NAME)
        self.processor.image_processor.min_pixels = self.min_pixels
        self.processor.image_processor.max_pixels = self.max_pixels
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Next2.5 VL GGUF model instance."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        _patch_qwen3_5_for_tt_device()

        # Load from the canonical (non-GGUF) repo so that the full config
        # (including vision_config) is available.  The GGUF file only carries a
        # text-only config which is insufficient for the VL model.
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self._PROCESSOR_NAME, **model_kwargs
        ).eval()

        self.config = model.config

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Next2.5 VL GGUF."""
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": self.sample_text},
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self._PROCESSOR_NAME)
        return self.config
