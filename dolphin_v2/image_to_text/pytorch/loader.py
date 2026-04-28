# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin-v2 model loader implementation for image-to-text document parsing.
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


class ModelVariant(StrEnum):
    """Available Dolphin-v2 model variants for image-to-text tasks."""

    DOLPHIN_V2 = "dolphin_v2"


def _patch_qwen2_5_vl_tolist():
    """Patch Qwen2.5-VL methods to move device tensors to CPU before .tolist() calls.

    TT device does not support eager tensor readback (.tolist()). These methods use
    .tolist() only for Python-level control flow, not for main computation.
    """
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m

    _orig_get_window_index = m.Qwen2_5_VisionTransformerPretrainedModel.get_window_index
    _orig_get_rope_index = m.Qwen2_5_VLModel.get_rope_index

    def _patched_rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw.cpu().tolist():
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
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids.to(grid_thw.device)].flatten(1)
        return rotary_pos_emb

    def _patched_get_window_index(self, grid_thw):
        device = grid_thw.device
        window_index, cu_window_seqlens = _orig_get_window_index(self, grid_thw.cpu())
        return window_index.to(device), cu_window_seqlens

    def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).cpu().tolist()
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
            input_ids.cpu() if input_ids is not None else None,
            image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw.cpu() if video_grid_thw is not None else None,
            second_per_grid_ts,
            attention_mask.cpu() if attention_mask is not None else None,
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


class ModelLoader(ForgeModel):
    """Dolphin-v2 model loader implementation for image-to-text document parsing."""

    _VARIANTS = {
        ModelVariant.DOLPHIN_V2: LLMModelConfig(
            pretrained_model_name="ByteDance/Dolphin-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOLPHIN_V2

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
            model="dolphin_v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance
        """
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        # transformers 5.x loads Qwen2VLImageProcessor as fast by default; use_fast=False
        # preserves the original slow processor behavior used when the checkpoint was saved.
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, use_fast=False, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Dolphin-v2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Dolphin-v2 model instance for image-to-text tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        _patch_qwen2_5_vl_tolist()

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Dolphin-v2 model.

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
                    {
                        "type": "text",
                        "text": "Parse the reading order of this document.",
                    },
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
        return inputs
