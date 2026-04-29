# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fara model loader implementation for vision-language tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
from .src.model import Wrapper


def _patch_qwen2_5_vl_for_tt_device():
    """Patch Qwen2.5 VL methods for TT device compatibility.

    Two classes of issues are fixed:

    1. .tolist() on TT tensors: get_rope_index, get_image_features, rot_pos_emb,
       and get_window_index call .tolist() on grid_thw / input_ids. TT device
       does not support eager tensor reads, so these tensors are moved to CPU
       before the call.

    2. torch.repeat_interleave tile-padding: the vision transformer forward uses
       repeat_interleave to build cu_seqlens. On TT device the output VALUE is
       tile-padded (e.g. 2204->2208), corrupting split_with_sizes in the vision
       attention. The fix clamps cu_seqlens to hidden_states.shape[0] inside
       Qwen2_5_VLVisionAttention.forward at the point of use, before
       lengths.tolist() reads the value back to Python.
    """
    try:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    except ImportError:
        return

    orig_get_rope = modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index
    orig_get_image = modeling_qwen2_5_vl.Qwen2_5_VLModel.get_image_features
    orig_rot_pos = modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb
    orig_get_window = modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.get_window_index
    orig_vis_attn_fwd = modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward

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

    def _patched_rot_pos(self, grid_thw):
        return orig_rot_pos(self, grid_thw.cpu() if grid_thw is not None else grid_thw)

    def _patched_get_window(self, grid_thw):
        return orig_get_window(self, grid_thw.cpu() if grid_thw is not None else grid_thw)

    def _patched_vis_attn_fwd(
        self,
        hidden_states,
        cu_seqlens=None,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        if cu_seqlens is not None:
            # TT repeat_interleave tile-pads tensor VALUES (e.g. 2204 -> 2208),
            # corrupting split_with_sizes. Clamp to actual seq len before
            # lengths.tolist() reads the value back to Python.
            cu_seqlens = torch.clamp(cu_seqlens, max=hidden_states.shape[0])
        return orig_vis_attn_fwd(
            self,
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index = _patched_get_rope
    modeling_qwen2_5_vl.Qwen2_5_VLModel.get_image_features = _patched_get_image
    modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.get_window_index = _patched_get_window
    modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = _patched_vis_attn_fwd


class ModelVariant(StrEnum):
    """Available Fara model variants for vision-language tasks."""

    FARA_7B = "7B"


class ModelLoader(ForgeModel):
    """Fara model loader implementation for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.FARA_7B: LLMModelConfig(
            pretrained_model_name="microsoft/Fara-7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FARA_7B

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

    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Fara",
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

        _patch_qwen2_5_vl_for_tt_device()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        # transformers 5.x no longer accepts use_cache in __init__; set via config
        model.config.text_config.use_cache = False
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
