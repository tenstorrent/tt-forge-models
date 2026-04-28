# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Glyph model loader implementation for multimodal conditional generation.
"""
import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from transformers.models.glm4v.modeling_glm4v import Glm4vModel, Glm4vVisionModel
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
from ....tools.utils import cast_input_to_type, get_file
from PIL import Image


class ModelVariant(StrEnum):
    """Available Glyph model variants for multimodal conditional generation."""

    GLYPH = "glyph"


class ModelLoader(ForgeModel):
    """Glyph model loader implementation for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.GLYPH: LLMModelConfig(
            pretrained_model_name="zai-org/Glyph",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLYPH

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="glyph",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_for_tt_device():
        """Patch GLM4V class methods for TT device execution.

        TT XLA moves all input tensors to TT device including metadata tensors
        like image_grid_thw. Several GLM4V methods use these tensors in Python
        control flow (.tolist(), iteration) or pass tensor elements to
        torch.arange(). TT device 0-d tensors cause INTERNAL error code 13 in
        those paths.

        We patch at class level (matching the Qwen3.5-VLM pattern) so that:
        - rot_pos_emb: converts grid_thw to Python ints via .cpu().tolist()
          before any torch.arange() call, avoiding TT-device dispatch.
        - Glm4vVisionModel.forward: moves grid_thw to CPU so that
          cu_seqlens.tolist() and rot_pos_emb both see CPU data.
        - get_image_features: moves image_grid_thw to CPU for split_sizes.tolist().
        - get_rope_index: moves input_ids/image_grid_thw/video_grid_thw to CPU
          for all .tolist() control-flow paths; returns position tensors to device.
        """
        _orig_rot_pos = Glm4vVisionModel.rot_pos_emb

        def _patched_rot_pos(self, grid_thw):
            # Move to CPU and convert to Python scalars so torch.arange() receives
            # Python ints and does not trigger TT device dispatch.
            grid_list = grid_thw.cpu().tolist()
            pos_ids = []
            for t, h, w in grid_list:
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
            # max_grid_size must be a Python int for rotary_pos_emb
            max_grid_size = max(max(h, w) for _, h, w in grid_list)
            rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
            rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
            return rotary_pos_emb, pos_ids

        Glm4vVisionModel.rot_pos_emb = _patched_rot_pos

        _orig_visual_fwd = Glm4vVisionModel.forward

        def _patched_visual_fwd(self, hidden_states, grid_thw, **kwargs):
            # Move grid_thw to CPU so cu_seqlens.tolist() in forward body works
            return _orig_visual_fwd(self, hidden_states, grid_thw.cpu(), **kwargs)

        Glm4vVisionModel.forward = _patched_visual_fwd

        _orig_get_img = Glm4vModel.get_image_features

        def _patched_get_img(self, pixel_values, image_grid_thw=None, **kwargs):
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.cpu()
            return _orig_get_img(self, pixel_values, image_grid_thw=image_grid_thw, **kwargs)

        Glm4vModel.get_image_features = _patched_get_img

        _orig_get_rope = Glm4vModel.get_rope_index

        def _patched_get_rope(self, input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None, **kwargs):
            orig_device = input_ids.device if input_ids is not None else "cpu"
            if input_ids is not None:
                input_ids = input_ids.cpu()
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.cpu()
            if video_grid_thw is not None:
                video_grid_thw = video_grid_thw.cpu()
            if attention_mask is not None:
                attention_mask = attention_mask.cpu()
            position_ids, mrope_position_deltas = _orig_get_rope(
                self,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                **kwargs,
            )
            return position_ids.to(orig_device), mrope_position_deltas.to(orig_device)

        Glm4vModel.get_rope_index = _patched_get_rope

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            use_fast=False,
            **kwargs,
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._patch_for_tt_device()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
