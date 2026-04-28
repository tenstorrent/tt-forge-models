# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DiffusionVL Qwen 2.5 VL model loader implementation for vision-language tasks.

hustvl/DiffusionVL-Qwen2.5VL-3B translates the autoregressive Qwen2.5-VL-3B
model into a diffusion vision language model using a block decoding strategy.
It ships custom modeling code via ``trust_remote_code`` and is exposed through
``AutoModelForCausalLM``.
"""
import requests
import torch
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from typing import Optional

_MODULE_NAME = "modeling_diffusionvl_qwen2_5_vl"


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
    """Available DiffusionVL Qwen 2.5 VL model variants."""

    DIFFUSIONVL_QWEN_2_5_VL_3B = "3B"


class ModelLoader(ForgeModel):
    """DiffusionVL Qwen 2.5 VL model loader implementation for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.DIFFUSIONVL_QWEN_2_5_VL_3B: LLMModelConfig(
            pretrained_model_name="hustvl/DiffusionVL-Qwen2.5VL-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIFFUSIONVL_QWEN_2_5_VL_3B

    sample_image_url = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DiffusionVL-Qwen2.5-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        # transformers 5.x loads Qwen2VLImageProcessor as a fast processor by
        # default, which is a breaking change.  Use use_fast=False to preserve
        # the original slow-processor behaviour.
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DiffusionVL Qwen 2.5 VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped DiffusionVL model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_cache": False,
            "trust_remote_code": True,
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        # transformers 5.x made two breaking changes to the custom model code:
        # 1. _tied_weights_keys must be a dict {target: source}, not a list.
        # 2. init_weights() calls tie_weights(recompute_mapping=False), but the
        #    custom tie_weights() doesn't accept **kwargs.
        # Patch both before loading so from_pretrained succeeds.
        _model_cls = get_class_from_dynamic_module(
            f"{_MODULE_NAME}.DiffusionVL_Qwen2_5_VL_ForConditionalGeneration",
            pretrained_model_name,
            trust_remote_code=True,
        )
        if isinstance(getattr(_model_cls, "_tied_weights_keys", None), list):
            _model_cls._tied_weights_keys = {
                "lm_head.weight": "model.embed_tokens.weight"
            }
        _orig_tie_weights = _model_cls.tie_weights
        if not getattr(_orig_tie_weights, "_kwargs_patched", False):

            def _tie_weights_compat(self, **kwargs):
                _orig_tie_weights(self)

            _tie_weights_compat._kwargs_patched = True
            _model_cls.tie_weights = _tie_weights_compat

        # 3. _merge_vision_text stub calls embed_tokens on input_ids that contain
        #    IMAGE_TOKEN_INDEX (-200), which is out of range.  Replace -200 with 0
        #    so the embedding lookup succeeds (sequence length stays unchanged so
        #    the attention_mask remains consistent).
        def _merge_vision_text_compat(self, input_ids, vision_features):
            safe_ids = input_ids.clone()
            safe_ids[safe_ids == -200] = 0
            return self.model.embed_tokens(safe_ids)

        _model_cls._merge_vision_text = _merge_vision_text_compat

        # 4. rot_pos_emb and get_window_index iterate over grid_thw using Python
        #    scalars (torch.arange(h), .tolist(), .item()).  On the TT device these
        #    readbacks fail with INTERNAL: Error code: 13.  Rewrite rot_pos_emb to
        #    use grid_thw.cpu().tolist(), and wrap get_window_index to move grid_thw
        #    to CPU before the call (returning window_index back on the original
        #    device so the subsequent hidden_states[window_index] indexing works).
        _vision_xfmr_cls = get_class_from_dynamic_module(
            f"{_MODULE_NAME}.DiffusionVL_Qwen2_5_VL_VisionTransformer",
            pretrained_model_name,
            trust_remote_code=True,
        )

        def _rot_pos_emb_compat(self, grid_thw):
            pos_ids = []
            for t, h, w in grid_thw.cpu().tolist():
                hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
                hpos_ids = hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
                wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
                wpos_ids = wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
                pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
            pos_ids = torch.cat(pos_ids, dim=0)
            max_grid_size = grid_thw[:, 1:].max()
            rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
            rotary_pos_emb = rotary_pos_emb_full[pos_ids.to(grid_thw.device)].flatten(1)
            return rotary_pos_emb

        _vision_xfmr_cls.rot_pos_emb = _rot_pos_emb_compat

        _orig_get_window_index = _vision_xfmr_cls.get_window_index
        if not getattr(_orig_get_window_index, "_cpu_patched", False):

            def _get_window_index_compat(self, grid_thw):
                device = grid_thw.device
                window_index, cu_window_seqlens = _orig_get_window_index(
                    self, grid_thw.cpu()
                )
                return window_index.to(device), cu_window_seqlens

            _get_window_index_compat._cpu_patched = True
            _vision_xfmr_cls.get_window_index = _get_window_index_compat

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DiffusionVL Qwen 2.5 VL model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        image = Image.open(
            BytesIO(requests.get(self.sample_image_url).content)
        ).convert("RGB")

        text = self.processor.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
