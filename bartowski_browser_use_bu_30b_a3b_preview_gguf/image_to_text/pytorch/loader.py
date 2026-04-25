# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski browser-use BU-30B-A3B-Preview GGUF model loader implementation for image to text.
"""

from contextlib import contextmanager

from transformers import (
    Qwen3VLMoeForConditionalGeneration,
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

# Save a reference to the original get_gguf_hf_weights_map before any other
# loaders install their patches. This module is imported early (alphabetically)
# so this captures the unpatched transformers function that we can call directly
# in the context manager, bypassing incompatible patch chains from other loaders.
import transformers.modeling_gguf_pytorch_utils as _gguf_utils_ref

_ORIG_GET_GGUF_HF_WEIGHTS_MAP = _gguf_utils_ref.get_gguf_hf_weights_map


def _patch_transformers_qwen3vlmoe_gguf():
    """Monkey-patch transformers to add qwen3vlmoe GGUF architecture support.

    The gguf library already knows about qwen3vlmoe tensor names, but
    transformers lacks the config mapping and architecture registration
    needed to load qwen3vlmoe GGUF checkpoints.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vlmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vlmoe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vlmoe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vlmoe"] = GGUFQwen2Converter

    _QWEN3VLMOE_TEXT_CONFIG_KEYS = {
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "vocab_size",
        "num_experts",
        "num_experts_per_tok",
    }

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vlmoe":
            config["model_type"] = "qwen3_vl_moe"
            # Move text backbone values into nested text_config so
            # Qwen3VLMoeConfig properly initializes the text model with the
            # GGUF dimensions.
            text_config = {}
            for key in list(config.keys()):
                if key in _QWEN3VLMOE_TEXT_CONFIG_KEYS:
                    text_config[key] = config.pop(key)
            if text_config:
                # Supply rope_scaling defaults matching Qwen3-VL-30B-A3B-Instruct
                # (mrope_section sums to head_dim/2=64, one entry per RoPE dim group).
                if "rope_scaling" not in text_config:
                    text_config["rope_scaling"] = {
                        "rope_type": "default",
                        "mrope_section": [24, 20, 20],
                        "mrope_interleaved": True,
                    }
                config["text_config"] = text_config
                # The GGUF file doesn't include vision config params so
                # vision_config.out_hidden_size defaults to 3584. Override it
                # to match the text hidden_size so the merger output matches
                # the text embedding dimension.
                if "hidden_size" in text_config:
                    config["vision_config"] = {
                        "out_hidden_size": text_config["hidden_size"]
                    }
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen3vlmoe_gguf()


@contextmanager
def _qwen3vlmoe_weights_map_patch():
    """Transiently install a get_gguf_hf_weights_map patch as the outermost wrapper.

    Applied inside load_model() so it is always the outermost wrapper. For
    Qwen3VLMoe models the top-level config stores num_hidden_layers inside
    text_config. We call _ORIG_GET_GGUF_HF_WEIGHTS_MAP directly to bypass
    incompatible patch chains from other loaders (which mix the 'processor'
    positional arg with 'model_type' keyword args).
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    prev = gguf_utils.get_gguf_hf_weights_map

    def patched(*args, **kwargs):
        hf_model = args[0] if args else kwargs.get("hf_model")
        # processor is args[1] when called from load_gguf_checkpoint
        processor = args[1] if len(args) > 1 else None
        # model_type and num_layers may be positional (recursive internal calls)
        # or keyword (our own top-level call).
        model_type = kwargs.get("model_type", args[2] if len(args) > 2 else None)
        num_layers = kwargs.get("num_layers", args[3] if len(args) > 3 else None)
        qual_name = kwargs.get("qual_name", args[4] if len(args) > 4 else "")
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type == "qwen3_vl_moe":
            if num_layers is None:
                try:
                    num_layers = hf_model.config.text_config.num_hidden_layers
                except AttributeError:
                    pass
            # Temporarily restore the original function at the module level so
            # that recursive calls inside _ORIG_GET_GGUF_HF_WEIGHTS_MAP (which
            # look up get_gguf_hf_weights_map by name in its own module) bypass
            # this wrapper and the incompatible patch chain from other loaders.
            gguf_utils.get_gguf_hf_weights_map = _ORIG_GET_GGUF_HF_WEIGHTS_MAP
            try:
                if processor is not None:
                    return _ORIG_GET_GGUF_HF_WEIGHTS_MAP(
                        hf_model,
                        processor,
                        model_type="qwen3vlmoe",
                        num_layers=num_layers,
                        qual_name=qual_name,
                    )
                return _ORIG_GET_GGUF_HF_WEIGHTS_MAP(
                    hf_model,
                    model_type="qwen3vlmoe",
                    num_layers=num_layers,
                    qual_name=qual_name,
                )
            finally:
                gguf_utils.get_gguf_hf_weights_map = patched
        return prev(*args, **kwargs)

    gguf_utils.get_gguf_hf_weights_map = patched
    try:
        yield
    finally:
        gguf_utils.get_gguf_hf_weights_map = prev


class ModelVariant(StrEnum):
    """Available bartowski browser-use BU-30B-A3B-Preview GGUF model variants for image to text."""

    BARTOWSKI_BROWSER_USE_BU_30B_A3B_PREVIEW_GGUF = (
        "browser_use_bu_30b_a3b_preview_gguf"
    )


class ModelLoader(ForgeModel):
    """bartowski browser-use BU-30B-A3B-Preview GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_BROWSER_USE_BU_30B_A3B_PREVIEW_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/browser-use_bu-30b-a3b-preview-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_BROWSER_USE_BU_30B_A3B_PREVIEW_GGUF

    GGUF_FILE = "browser-use_bu-30b-a3b-preview-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="bartowski browser-use BU-30B-A3B-Preview GGUF",
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
        self.processor = AutoProcessor.from_pretrained("browser-use/bu-30b-a3b-preview")

        with _qwen3vlmoe_weights_map_patch():
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
