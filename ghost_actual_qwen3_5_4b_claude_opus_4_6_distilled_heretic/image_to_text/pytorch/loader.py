# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ghost-Actual Qwen3.5-4B Claude Opus 4.6 Distilled Heretic model loader
implementation for image to text.
"""

from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
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


def _patch_qwen3_5_for_tt_device():
    """Patch Qwen3.5 VL methods that call .tolist() on device tensors.

    TT device does not support eager tensor reads — any .tolist() on a TT
    tensor triggers a device sync that fails with Error code: 13. Move the
    small integer metadata tensors (grid_thw, input_ids) to CPU before the
    .tolist() calls; move computed outputs (position_ids, rope_deltas) back
    to the original device so the language model path stays on TT device.
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
    """Available Ghost-Actual Qwen3.5-4B Claude Opus 4.6 Distilled Heretic variants for image to text."""

    QWEN_3_5_4B_CLAUDE_OPUS_4_6_DISTILLED_HERETIC = (
        "4b_claude_opus_4_6_distilled_heretic"
    )


class ModelLoader(ForgeModel):
    """Ghost-Actual Qwen3.5-4B Claude Opus 4.6 Distilled Heretic model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_4B_CLAUDE_OPUS_4_6_DISTILLED_HERETIC: LLMModelConfig(
            pretrained_model_name="ghost-actual/Qwen3.5-4B-Claude-Opus-4.6-Distilled-heretic",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_4B_CLAUDE_OPUS_4_6_DISTILLED_HERETIC

    # Standard pixel limits for Qwen VL models to stay within hardware L1 budget
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ghost_actual_qwen3_5_4b_claude_opus_4_6_distilled_heretic",
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

        # The model repo does not ship preprocessor_config.json, so AutoProcessor
        # cannot auto-detect the image processor. Construct it manually: qwen3_5
        # maps to Qwen2VLImageProcessor in AutoImageProcessor, and Qwen3VLProcessor
        # is the processor class declared in the model's tokenizer_config.json.
        # Qwen3.5 vision config uses patch_size=16 (vs Qwen2-VL default of 14).
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        image_processor = Qwen2VLImageProcessor(
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            patch_size=16,
        )
        video_processor = Qwen3VLVideoProcessor()
        self.processor = Qwen3VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=tokenizer.chat_template,
        )

        _patch_qwen3_5_for_tt_device()

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
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
