# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-OneVision-1.5 model loader implementation for multimodal conditional generation.
"""

import torch
import transformers.cache_utils as _cache_utils
import transformers.modeling_rope_utils as _rope_utils
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.configuration_utils import PretrainedConfig as _PretrainedConfig
from typing import Optional

# transformers 5.x removed SlidingWindowCache; add a stub so the model's
# trust_remote_code module (which imports it for isinstance checks) can load.
if not hasattr(_cache_utils, "SlidingWindowCache"):

    class SlidingWindowCache:
        pass

    _cache_utils.SlidingWindowCache = SlidingWindowCache

# transformers 5.x removed pad_token_id from PretrainedConfig; the remote
# model code reads config.pad_token_id to set the embedding padding index.
if not hasattr(_PretrainedConfig, "pad_token_id"):
    _PretrainedConfig.pad_token_id = None

# transformers 5.x removed the "default" key from ROPE_INIT_FUNCTIONS; the
# remote model code falls through to rope_type="default" when no rope_scaling
# is configured.  Provide a vanilla RoPE implementation under that key.
if "default" not in _rope_utils.ROPE_INIT_FUNCTIONS:

    def _default_rope_init(config, device=None, **kwargs):
        base = config.rope_theta
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim
            )
        )
        return inv_freq, 1.0

    _rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _patch_model_for_tt(model):
    """Patch control-flow methods that Python-iterate over TT device tensors.

    rot_pos_emb: use tolist() for Python control flow so pos_ids are created
    directly on the TT device, avoiding a cross-device gather.

    get_image_features: pass image_grid_thw as a CPU tensor so that cu_seqlens
    inside the visual encoder forward stays on CPU, enabling .item() calls.

    get_rope_index: move all inputs to CPU (only integer metadata) before the
    original implementation's tolist()/iteration, then restore device.
    """

    # rot_pos_emb: tolist() gives Python ints for the loop; pos_ids stay on the
    # TT device so the final gather (rotary_pos_emb_full[pos_ids]) is device-local.
    VisionClass = type(model.model.visual)

    def _patched_rot_pos_emb(self, grid_thw):
        grid_list = grid_thw.tolist()
        pos_ids = []
        device = self.rotary_pos_emb.inv_freq.device
        for t, h, w in grid_list:
            hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
            hpos = hpos.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
            wpos = wpos.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_list)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    VisionClass.rot_pos_emb = _patched_rot_pos_emb

    # get_image_features: keep image_grid_thw on CPU so cu_seqlens stays CPU,
    # enabling .item() in the visual encoder's patch-merge segment loop.
    MainModelClass = type(model.model)

    def _patched_get_image_features(self, pixel_values, image_grid_thw=None):
        pixel_values = pixel_values.type(self.visual.dtype)
        grid_cpu = image_grid_thw.cpu() if image_grid_thw is not None else None
        return self.visual(pixel_values, grid_thw=grid_cpu)

    MainModelClass.get_image_features = _patched_get_image_features

    # get_rope_index: passes integer metadata tensors as CPU to the original
    # implementation (which calls tolist()/iteration), then restores device.
    _orig_get_rope_index = MainModelClass.get_rope_index

    def _patched_get_rope_index(self, input_ids=None, image_grid_thw=None,
                                video_grid_thw=None, attention_mask=None):
        orig_device = (
            input_ids.device if input_ids is not None
            else image_grid_thw.device if image_grid_thw is not None
            else None
        )
        position_ids, rope_deltas = _orig_get_rope_index(
            self,
            input_ids=input_ids.cpu() if input_ids is not None else None,
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
        )
        if orig_device is not None:
            position_ids = position_ids.to(orig_device)
            rope_deltas = rope_deltas.to(orig_device)
        return position_ids, rope_deltas

    MainModelClass.get_rope_index = _patched_get_rope_index


class ModelVariant(StrEnum):
    """Available LLaVA-OneVision-1.5 model variants."""

    LLAVA_ONEVISION_1_5_4B_BASE = "4B_Base"
    LLAVA_ONEVISION_1_5_4B_INSTRUCT = "4B_Instruct"
    LLAVA_ONEVISION_1_5_8B_INSTRUCT = "8B_Instruct"


class ModelLoader(ForgeModel):
    """LLaVA-OneVision-1.5 model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_ONEVISION_1_5_4B_BASE: ModelConfig(
            pretrained_model_name="lmms-lab/LLaVA-OneVision-1.5-4B-Base",
        ),
        ModelVariant.LLAVA_ONEVISION_1_5_4B_INSTRUCT: ModelConfig(
            pretrained_model_name="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        ),
        ModelVariant.LLAVA_ONEVISION_1_5_8B_INSTRUCT: ModelConfig(
            pretrained_model_name="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_ONEVISION_1_5_8B_INSTRUCT

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

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-OneVision-1.5 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-OneVision-1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-OneVision-1.5 model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = "auto"
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        _patch_model_for_tt(model)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-OneVision-1.5."""
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
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
