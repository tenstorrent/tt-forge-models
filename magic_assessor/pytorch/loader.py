# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MagicAssessor model loader implementation for vision-language tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional


def _patch_qwen2_5_vl_for_tt():
    """Patch Qwen2.5-VL int64 D2H ops that fail on TT silicon (INTERNAL error 13).

    On TT silicon, int64/int32 device-to-host transfers fail; float32 D2H works.
    Use float32 bridge (tensor.float().cpu().long()) for all int metadata tensors
    that require Python-level value access (tolist(), indexing, control flow).
    Applied inside load_model() to avoid double-patching during test collection.
    """
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m

    if getattr(m.Qwen2_5_VisionTransformerPretrainedModel, "_tt_int_patch", False):
        return
    m.Qwen2_5_VisionTransformerPretrainedModel._tt_int_patch = True

    _orig_get_window_index = m.Qwen2_5_VisionTransformerPretrainedModel.get_window_index
    _orig_get_rope_index = m.Qwen2_5_VLModel.get_rope_index

    def _patched_rot_pos_emb(self, grid_thw):
        pos_ids = []
        # Float32 bridge: int64 D2H fails on TT silicon (INTERNAL error 13)
        grid_thw_cpu = grid_thw.float().cpu().long()
        for t, h, w in grid_thw_cpu.tolist():
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw_cpu[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids.to(grid_thw.device)].flatten(1)
        return rotary_pos_emb

    def _patched_get_window_index(self, grid_thw):
        return _orig_get_window_index(self, grid_thw.float().cpu().long())

    def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_grid_thw_cpu = image_grid_thw.float().cpu().long()
        split_sizes = (image_grid_thw_cpu.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds
        return vision_outputs

    def _patched_get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
        **kwargs,
    ):
        device = input_ids.device if input_ids is not None else None
        position_ids, rope_deltas = _orig_get_rope_index(
            self,
            input_ids.float().cpu().long() if input_ids is not None else None,
            image_grid_thw.float().cpu().long() if image_grid_thw is not None else None,
            video_grid_thw.float().cpu().long() if video_grid_thw is not None else None,
            second_per_grid_ts,
            attention_mask.float().cpu().to(attention_mask.dtype) if attention_mask is not None else None,
            **kwargs,
        )
        if device is not None:
            position_ids = position_ids.to(device)
            rope_deltas = rope_deltas.to(device)
        return position_ids, rope_deltas

    m.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb = _patched_rot_pos_emb
    m.Qwen2_5_VisionTransformerPretrainedModel.get_window_index = _patched_get_window_index
    m.Qwen2_5_VLModel.get_image_features = _patched_get_image_features
    m.Qwen2_5_VLModel.get_rope_index = _patched_get_rope_index


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

        _patch_qwen2_5_vl_for_tt()

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
