# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher Huihui Qwen3-VL 8B Instruct Abliterated GGUF model loader
implementation for image to text.
"""

from transformers import (
    AutoConfig,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
)


def _patch_qwen3vl_support():
    """Register qwen3vl GGUF architecture and patch weights-map for qwen3_vl HF model type.

    - qwen3vl is not in GGUF_SUPPORTED_ARCHITECTURES in transformers 5.x.
    - get_gguf_hf_weights_map uses hf_model.config.model_type ('qwen3_vl') to
      look up in gguf-py MODEL_ARCH_NAMES, which only has 'qwen3vl' (no underscore).
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    if "qwen3vl" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
        if "qwen3" in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
            GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = dict(
                GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3"]
            )

    orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
            if num_layers is None:
                cfg = hf_model.config
                text_cfg = getattr(cfg, "text_config", cfg)
                num_layers = getattr(text_cfg, "num_hidden_layers", None)
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    if not getattr(_gguf_utils.get_gguf_hf_weights_map, "_qwen3vl_patched", False):
        _patched_get_gguf_hf_weights_map._qwen3vl_patched = True
        _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen3vl_support()


def _patch_qwen3vl_for_tt_device():
    """Patch Qwen3VL methods that call .tolist() on TT device tensors.

    grid_thw and related tensors land on the XLA device; .tolist() triggers
    a D2H transfer that raises INTERNAL error 13.  Move those tensors to CPU
    before the calls that need Python ints, then move outputs back.
    """
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    if getattr(modeling_qwen3_vl, "_tt_tolist_patched", False):
        return

    orig_fast_pos = modeling_qwen3_vl.Qwen3VLVisionModel.fast_pos_embed_interpolate
    orig_rot_pos = modeling_qwen3_vl.Qwen3VLVisionModel.rot_pos_emb
    orig_get_rope = modeling_qwen3_vl.Qwen3VLModel.get_rope_index
    orig_get_image = modeling_qwen3_vl.Qwen3VLModel.get_image_features

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

    modeling_qwen3_vl.Qwen3VLVisionModel.fast_pos_embed_interpolate = _patched_fast_pos
    modeling_qwen3_vl.Qwen3VLVisionModel.rot_pos_emb = _patched_rot_pos
    modeling_qwen3_vl.Qwen3VLModel.get_rope_index = _patched_get_rope
    modeling_qwen3_vl.Qwen3VLModel.get_image_features = _patched_get_image
    modeling_qwen3_vl._tt_tolist_patched = True


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
    """Available Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF variants for image to text."""

    HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF = (
        "8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: "Huihui-Qwen3-VL-8B-Instruct-abliterated.Q4_K_M.gguf",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
        )
        # Limit image patch count to prevent excessive sequence lengths.
        self.processor.image_processor.min_pixels = 56 * 56
        self.processor.image_processor.max_pixels = 13 * 28 * 1280

        # Use the unquantized base model config so model architecture dimensions are correct.
        # Explicit config bypasses qwen3vl architecture check in GGUF config loading;
        # tensor loading still calls get_gguf_hf_weights_map (patched at import time).
        config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            **model_kwargs,
        )
        model.eval()
        _patch_qwen3vl_for_tt_device()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
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
