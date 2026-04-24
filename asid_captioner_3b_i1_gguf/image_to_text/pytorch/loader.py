# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ASID Captioner 3B i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen2VLForConditionalGeneration,
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

_TRANSFORMERS_LOAD_GGUF = None


def _get_real_load_gguf_checkpoint():
    """Load a fresh, unpatched load_gguf_checkpoint from the transformers source file.

    Other GGUF loaders monkey-patch transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint
    with wrappers that have fixed signatures (gguf_path, return_tensors=False) stored as
    module-level globals rather than closures, making closure traversal impossible.
    Loading a fresh copy from source bypasses all patches while still seeing our
    GGUF_CONFIG_MAPPING modifications (shared dict via sys.modules).

    Also patches get_gguf_hf_weights_map in the fresh module to handle Qwen2VL:
    - maps model_type "qwen2_vl" → "qwen2vl" (the gguf arch name)
    - resolves num_hidden_layers via text_config for VL models that nest it there
    """
    import importlib.util

    spec = importlib.util.find_spec("transformers.modeling_gguf_pytorch_utils")
    fresh_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh_module)

    _orig_get_weights_map = fresh_module.get_gguf_hf_weights_map

    def _qwen2vl_get_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None or num_layers is None:
            cfg = getattr(hf_model, "config", None)
            if cfg is not None:
                if model_type is None:
                    raw = getattr(cfg, "model_type", None)
                    model_type = "qwen2vl" if raw == "qwen2_vl" else raw
                if num_layers is None:
                    if hasattr(cfg, "text_config"):
                        num_layers = cfg.text_config.num_hidden_layers
                    elif hasattr(cfg, "num_hidden_layers"):
                        num_layers = cfg.num_hidden_layers
        return _orig_get_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    fresh_module.get_gguf_hf_weights_map = _qwen2vl_get_weights_map

    return fresh_module.load_gguf_checkpoint


def _register_qwen2vl_gguf_architecture():
    """Register qwen2vl in transformers GGUF config mapping and save the real load function."""
    global _TRANSFORMERS_LOAD_GGUF

    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
    )

    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    if "qwen2vl" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["qwen2vl"] = {
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

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    # Load the real transformers function from source, bypassing all monkey-patches.
    if _TRANSFORMERS_LOAD_GGUF is None:
        _TRANSFORMERS_LOAD_GGUF = _get_real_load_gguf_checkpoint()


_register_qwen2vl_gguf_architecture()


def _apply_qwen2vl_load_patch():
    """Install a load_gguf_checkpoint wrapper that handles qwen2vl model_type remapping.

    Installs immediately before from_pretrained to be the outermost wrapper.
    Calls the real transformers function directly (bypassing other models'
    broken intermediate patches) then remaps qwen2vl -> qwen2_vl.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    real_fn = _TRANSFORMERS_LOAD_GGUF or _get_real_load_gguf_checkpoint()

    def _qwen2vl_patched(*args, **kwargs):
        result = real_fn(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            config["model_type"] = "qwen2_vl"
        return result

    gguf_utils.load_gguf_checkpoint = _qwen2vl_patched
    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _qwen2vl_patched


class ModelVariant(StrEnum):
    """Available ASID Captioner 3B i1 GGUF model variants for image to text."""

    ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF = "3b_i1_Q4_K_M_gguf"


class ModelLoader(ForgeModel):
    """ASID Captioner 3B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/ASID-Captioner-3B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF

    GGUF_FILE = "ASID-Captioner-3B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ASID Captioner 3B i1 GGUF",
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
        self.processor = AutoProcessor.from_pretrained(
            "AudioVisual-Caption/ASID-Captioner-3B"
        )

        # Re-apply patch right before from_pretrained so our wrapper is the
        # outermost and calls the real transformers function directly.
        _apply_qwen2vl_load_patch()

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
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
