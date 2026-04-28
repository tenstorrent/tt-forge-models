# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FireRed-OCR GGUF model loader implementation for image-to-text tasks.

FireRed-OCR is an OCR model based on Qwen3-VL-2B-Instruct. This loader
consumes the GGUF-quantized checkpoints published at
https://huggingface.co/mradermacher/FireRed-OCR-GGUF.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
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


def _patch_transformers_qwen3vl_gguf():
    """Monkey-patch transformers to add qwen3vl GGUF architecture support.

    The qwen3vl GGUF format stores Qwen3-VL LLM weights only (vision encoder
    is not included). Transformers 5.x lacks GGUF loading support for the
    qwen3vl architecture. This patch:
    1. Registers qwen3vl as a supported GGUF architecture (same config
       fields as qwen3).
    2. Patches load_gguf_checkpoint to set model_type = "qwen3_vl".
    3. Patches get_gguf_hf_weights_map to handle the qwen3_vl model type
       and the composite VL config that lacks a top-level num_hidden_layers.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        load_gguf_checkpoint as _orig_load,
        get_gguf_hf_weights_map as _orig_weights_map,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3vl as a supported architecture (LLM fields same as qwen3)
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    # 2. Register tokenizer converter (BPE-based, same as qwen3)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    # 3. Patch load_gguf_checkpoint to set model_type and restructure config.
    #    Qwen3VLConfig expects LLM params nested under text_config, not flat.
    _LLM_FIELDS = {
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
    }

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
            # Move flat LLM params into text_config so Qwen3VLConfig picks
            # up the 2B architecture instead of the 7B defaults.
            text_cfg = {k: config.pop(k) for k in _LLM_FIELDS if k in config}
            if text_cfg:
                config["text_config"] = text_cfg
                # vision_config.out_hidden_size must match LLM hidden_size so
                # the visual merger output dimension is consistent.
                llm_hidden = text_cfg.get("hidden_size")
                if llm_hidden is not None:
                    config["vision_config"] = {"out_hidden_size": llm_hidden}
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # 4. Patch get_gguf_hf_weights_map to handle qwen3_vl composite config.
    #    Qwen3VLConfig has no num_hidden_layers at the top level (it is nested
    #    in text_config). Also maps "qwen3_vl" -> "qwen3vl" for the gguf-py
    #    MODEL_ARCH_NAMES lookup.
    #
    #    For qwen3vl GGUF files: only LLM weights are present; the vision
    #    encoder is not. Qwen3VLModel processes visual before language_model in
    #    named_children(), so merger.norm would steal the output_norm mapping.
    #    We bypass the full traversal and map language_model directly.
    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
        if num_layers is None:
            num_layers = getattr(hf_model.config, "num_hidden_layers", None)
            if num_layers is None:
                text_cfg = getattr(hf_model.config, "text_config", None)
                num_layers = (
                    getattr(text_cfg, "num_hidden_layers", 1) if text_cfg else 1
                )

        # For qwen3vl VL models at the top-level call, build the map only from
        # model.language_model to avoid the visual.merger.norm collision with
        # the language model's output_norm.
        if model_type == "qwen3vl" and qual_name == "":
            model_sub = getattr(hf_model, "model", None)
            if model_sub is not None and hasattr(model_sub, "language_model"):
                return _orig_weights_map(
                    model_sub.language_model,
                    processor,
                    model_type=model_type,
                    num_layers=num_layers,
                    qual_name="model.language_model.",
                )

        return _orig_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils

    for mod in (tok_auto, config_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_transformers_qwen3vl_gguf()


def _patch_qwen3vl_for_tt_device():
    """Patch Qwen3 VL methods that call .tolist() on device tensors.

    The test runner moves all input tensors to TT device, but the VisionModel
    and get_rope_index methods call .tolist() on grid_thw / input_ids tensors for
    Python control flow. TT device does not support eager tensor reads — they
    trigger a device sync that fails with Error code: 13. Moving these metadata
    tensors to CPU before the .tolist() calls avoids the sync while keeping all
    actual vision and language computations on TT device.
    """
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except ImportError:
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


class ModelVariant(StrEnum):
    """Available FireRed-OCR GGUF model variants for image-to-text."""

    FIRERED_OCR_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """FireRed-OCR GGUF model loader implementation for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.FIRERED_OCR_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/FireRed-OCR-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FIRERED_OCR_Q4_K_M

    GGUF_FILE = "FireRed-OCR.Q4_K_M.gguf"

    # GGUF repos do not ship a processor; use the base model.
    BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

    # Standard pixel limits for Qwen VL models to stay within hardware L1 budget
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="FireRed-OCR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)
        self.processor.image_processor.min_pixels = self.min_pixels
        self.processor.image_processor.max_pixels = self.max_pixels

        _patch_qwen3vl_for_tt_device()

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)
            self.processor.image_processor.min_pixels = self.min_pixels
            self.processor.image_processor.max_pixels = self.max_pixels

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Extract the text from this image."},
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
