# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Apexlearn BECoach-Qwen3.5-4B 4-bit VLM MLX model loader implementation for image-text-to-text generation.
"""

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
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


def _patch_qwen35_for_tt_device():
    """Patch Qwen3.5 VLM methods that call .tolist() on device tensors.

    The test runner moves all input tensors to TT device, but the VisionEncoder
    and get_rope_index methods call .tolist() on grid_thw / input_ids tensors for
    Python control flow. TT device does not support eager tensor reads (they would
    trigger a device sync that fails with Error code: 13). Moving these tensors to
    CPU before the .tolist() calls avoids the sync while keeping all actual
    computations on TT device.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        return

    orig_fast_pos = modeling_qwen3_5.Qwen3_5VisionModel.fast_pos_embed_interpolate
    orig_rot_pos = modeling_qwen3_5.Qwen3_5VisionModel.rot_pos_emb
    orig_get_rope = modeling_qwen3_5.Qwen3_5Model.get_rope_index

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
    ):
        return orig_get_rope(
            self,
            input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
        )

    modeling_qwen3_5.Qwen3_5VisionModel.fast_pos_embed_interpolate = _patched_fast_pos
    modeling_qwen3_5.Qwen3_5VisionModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen3_5.Qwen3_5Model.get_rope_index = _patched_get_rope


class ModelVariant(StrEnum):
    """Available Apexlearn BECoach-Qwen3.5-4B 4-bit VLM MLX model variants."""

    BECOACH_QWEN3_5_4B_4BIT_VLM_MLX = "BECoach_Qwen3_5_4B_4bit_vlm_mlx"


class ModelLoader(ForgeModel):
    """Apexlearn BECoach-Qwen3.5-4B 4-bit VLM MLX model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.BECOACH_QWEN3_5_4B_4BIT_VLM_MLX: LLMModelConfig(
            pretrained_model_name="apexlearn/BECoach-Qwen3.5-4B-4bit-vlm-mlx",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BECOACH_QWEN3_5_4B_4BIT_VLM_MLX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Apexlearn BECoach-Qwen3.5-4B 4-bit VLM MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        model_kwargs["config"] = config

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        _patch_qwen35_for_tt_device()

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

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
