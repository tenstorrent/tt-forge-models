# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GELab-Zero-4B-preview GGUF model loader implementation for image to text.
"""

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.models.qwen3_vl import modeling_qwen3_vl as _qwen3vl_mod
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

# ---------------------------------------------------------------------------
# Patch: register "qwen3vl" as a supported GGUF architecture.
#
# transformers (5.x) only knows about "qwen3" and "qwen3_moe" GGUF archs.
# Qwen3VL GGUF files identify themselves with general.architecture=qwen3vl,
# which causes load_gguf_checkpoint to raise ValueError before reading any
# weights.  We also need to fix get_gguf_hf_weights_map, which looks up the
# HF model_type ("qwen3_vl") in gguf-py's MODEL_ARCH_NAMES and expects to
# find "qwen3vl", and which tries to read config.num_hidden_layers directly
# even though Qwen3VLConfig nests it under text_config.
# ---------------------------------------------------------------------------
def _patch_qwen3vl_gguf_support():
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

    if "qwen3vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        # Reuse the qwen3 (text-only) config field mapping as a stand-in.
        # The extracted config dict is never used: we pass an explicit
        # Qwen3VLConfig to from_pretrained(), so only the architecture guard
        # needs to pass.
        GGUF_CONFIG_MAPPING.setdefault("qwen3vl", GGUF_CONFIG_MAPPING["qwen3"])
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    orig_fn = _gguf_utils.get_gguf_hf_weights_map
    if getattr(orig_fn, "_qwen3vl_patched", False):
        return

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        # model_type may already be "qwen3_vl" if a prior patch read it from
        # hf_model.config.model_type before calling us (e.g. momix_44 does
        # this), so we must check for the HF name regardless of None.
        effective_mt = model_type or (
            getattr(getattr(hf_model, "config", None), "model_type", "")
        )
        if effective_mt in ("qwen3_vl", "qwen3_vl_text"):
            # gguf-py uses "qwen3vl" as the arch string, not "qwen3_vl"
            model_type = "qwen3vl"
            if num_layers is None and hasattr(hf_model, "config"):
                # num_hidden_layers lives in text_config for composite configs
                text_cfg = getattr(hf_model.config, "text_config", hf_model.config)
                num_layers = getattr(text_cfg, "num_hidden_layers", None)
        return orig_fn(hf_model, processor, model_type, num_layers, qual_name)

    _patched_get_gguf_hf_weights_map._qwen3vl_patched = True
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen3vl_gguf_support()


# ---------------------------------------------------------------------------
# Patch: move grid_thw / input_ids metadata to CPU for .tolist() control flow.
#
# TT device does not support eager Python-side tensor readback (.tolist()).
# Four Qwen3VL methods call .tolist() on grid_thw or input_ids that may be
# on the TT device, causing RuntimeError: Bad StatusOr access INTERNAL:13.
# Only the control-flow metadata is moved; main computation stays on device.
# ---------------------------------------------------------------------------
def _patch_qwen3vl_tolist():
    _VisionModel = _qwen3vl_mod.Qwen3VLVisionModel
    _Model = _qwen3vl_mod.Qwen3VLModel

    if getattr(_VisionModel.fast_pos_embed_interpolate, "_tolist_patched", False):
        return

    _orig_fast_pos = _VisionModel.fast_pos_embed_interpolate

    def _fast_pos(self, grid_thw):
        return _orig_fast_pos(self, grid_thw.cpu())

    _fast_pos._tolist_patched = True
    _VisionModel.fast_pos_embed_interpolate = _fast_pos

    _orig_rot_pos = _VisionModel.rot_pos_emb

    def _rot_pos(self, grid_thw):
        return _orig_rot_pos(self, grid_thw.cpu())

    _rot_pos._tolist_patched = True
    _VisionModel.rot_pos_emb = _rot_pos

    _orig_get_image_feat = _Model.get_image_features

    def _get_image_feat(self, pixel_values, image_grid_thw=None, **kwargs):
        # prod(-1).tolist() needs CPU; pixel_values stays on device
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.cpu()
        return _orig_get_image_feat(self, pixel_values, image_grid_thw, **kwargs)

    _get_image_feat._tolist_patched = True
    _Model.get_image_features = _get_image_feat

    _orig_get_rope = _Model.get_rope_index

    def _get_rope(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        orig_device = input_ids.device if input_ids is not None else None
        if input_ids is not None:
            input_ids = input_ids.cpu()
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.cpu()
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.cpu()
        if attention_mask is not None:
            attention_mask = attention_mask.cpu()
        position_ids, mrope_deltas = _orig_get_rope(
            self,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            **kwargs,
        )
        if orig_device is not None:
            position_ids = position_ids.to(orig_device)
            mrope_deltas = mrope_deltas.to(orig_device)
        return position_ids, mrope_deltas

    _get_rope._tolist_patched = True
    _Model.get_rope_index = _get_rope


_patch_qwen3vl_tolist()


class ModelVariant(StrEnum):
    """Available GELab-Zero-4B-preview GGUF model variants for image to text."""

    GELAB_ZERO_4B_PREVIEW_Q4_K_M_GGUF = "4B_PREVIEW_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """GELab-Zero-4B-preview GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.GELAB_ZERO_4B_PREVIEW_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="Mungert/GELab-Zero-4B-preview-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GELAB_ZERO_4B_PREVIEW_Q4_K_M_GGUF

    GGUF_FILE = "GELab-Zero-4B-preview-q4_k_m.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GELab-Zero-4B-preview GGUF",
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
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        # Constrain input resolution so the demo image (~1376×2048) produces a
        # manageable number of visual patches rather than 11K+.
        self.processor.image_processor.min_pixels = 56 * 56
        self.processor.image_processor.max_pixels = 13 * 28 * 1280

        # Pass the 4B base config explicitly so from_pretrained skips GGUF
        # config extraction (qwen3vl is not in transformers' GGUF config
        # mapping), while still loading weights from the GGUF file.
        # ignore_mismatched_sizes=True: gguf-py's qwen3vl tensor name map
        # incorrectly maps a [2560] text-model tensor to
        # model.visual.merger.norm.weight (which expects [1024]); allow that
        # one weight to be skipped rather than aborting the entire load.
        config = Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            **model_kwargs,
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
