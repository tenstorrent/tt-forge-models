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
    """Patch Qwen2.5-VL for TT silicon: route int64 metadata through CPU.

    Strategy:
    - load_inputs casts image_grid_thw to float32 so the device tensor is float32
    - get_image_features does one float32 D2H (.cpu().long()) and caches the
      CPU int64 result on self.visual._tt_grid_thw_cpu before calling self.visual
    - rot_pos_emb and get_window_index read the cache (no additional D2H)
    - VisionTransformer.forward is replaced to use the CPU int64 for
      repeat_interleave and move CPU index tensors to device before indexing
    - get_rope_index receives image_grid_thw as CPU int64 (from the cache),
      input_ids stays as-is (int64 device, embedding/scatter ops work on TT)
    Applied inside load_model() to avoid double-patching during test collection.
    """
    import torch.nn.functional as F
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    if getattr(m.Qwen2_5_VisionTransformerPretrainedModel, "_tt_int_patch", False):
        return
    m.Qwen2_5_VisionTransformerPretrainedModel._tt_int_patch = True

    _orig_get_window_index = m.Qwen2_5_VisionTransformerPretrainedModel.get_window_index
    _orig_get_rope_index = m.Qwen2_5_VLModel.get_rope_index

    def _patched_rot_pos_emb(self, grid_thw):
        pos_ids = []
        # Use CPU int64 cached by _patched_get_image_features (no D2H needed here)
        grid_thw_cpu = getattr(self, "_tt_grid_thw_cpu", None)
        if grid_thw_cpu is None:
            grid_thw_cpu = grid_thw.cpu().long()
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
        # pos_ids.to(device): int64 H2D works; gather with int64 indices works like embedding
        rotary_pos_emb = rotary_pos_emb_full[pos_ids.to(rotary_pos_emb_full.device)].flatten(1)
        return rotary_pos_emb

    def _patched_get_window_index(self, grid_thw):
        # Use cached CPU int64 (no D2H needed)
        grid_thw_cpu = getattr(self, "_tt_grid_thw_cpu", None)
        if grid_thw_cpu is None:
            grid_thw_cpu = grid_thw.cpu().long()
        return _orig_get_window_index(self, grid_thw_cpu)

    def _patched_vt_forward(self, hidden_states, grid_thw, **kwargs):
        """Replaces VisionTransformerPretrainedModel.forward with TT-safe indexing.

        All int64 metadata stays on CPU; only int64 H2D transfers and gather ops
        (both supported on TT silicon like embedding lookup) are on device.
        """
        # Cached CPU int64 set by _patched_get_image_features before this call
        grid_thw_cpu = getattr(self, "_tt_grid_thw_cpu", None)
        if grid_thw_cpu is None:
            grid_thw_cpu = grid_thw.cpu().long()

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        device = hidden_states.device
        # unique_consecutive and tensor construction stay on CPU — TT silicon
        # implements unique_consecutive via tiled reductions which can read stale
        # padding cells and produce off-by-spatial_merge_unit results.
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_window_seqlens = cu_window_seqlens.to(device)

        seq_len, _ = hidden_states.size()
        # window_index is CPU int64; move to device for TT gather (int64 H2D + gather)
        window_index_d = window_index.to(device)
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index_d, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index_d, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # CPU int64 for repeat_interleave avoids float32-repeats dispatch failure
        cu_seqlens = torch.repeat_interleave(
            grid_thw_cpu[:, 1] * grid_thw_cpu[:, 2], grid_thw_cpu[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(device)

        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)
        # argsort on CPU; gather on device via int64 H2D
        reverse_indices = torch.argsort(window_index).to(device)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        # One float32 D2H to get CPU int64; cache on visual for rot_pos_emb / get_window_index
        self.visual._tt_grid_thw_cpu = image_grid_thw.cpu().long()
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        split_sizes = (self.visual._tt_grid_thw_cpu.prod(-1) // self.visual.spatial_merge_size**2).tolist()
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
        # image_grid_thw is float32 (cast in load_inputs); float32 D2H to CPU int64
        # input_ids stays as-is: embedding/comparison ops work on TT with int64
        position_ids, rope_deltas = _orig_get_rope_index(
            self,
            input_ids,
            image_grid_thw.cpu().long() if image_grid_thw is not None else None,
            video_grid_thw.cpu().long() if video_grid_thw is not None else None,
            second_per_grid_ts,
            attention_mask,
            **kwargs,
        )
        if device is not None:
            position_ids = position_ids.to(device)
            rope_deltas = rope_deltas.to(device)
        return position_ids, rope_deltas

    m.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb = _patched_rot_pos_emb
    m.Qwen2_5_VisionTransformerPretrainedModel.get_window_index = _patched_get_window_index
    m.Qwen2_5_VisionTransformerPretrainedModel.forward = _patched_vt_forward
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

        # TT silicon cannot compile int64 ops; cast grid_thw to float32 so the
        # patched VL methods can use float32 D2H (.cpu()) to get CPU values.
        inputs["image_grid_thw"] = inputs["image_grid_thw"].float()

        return inputs
