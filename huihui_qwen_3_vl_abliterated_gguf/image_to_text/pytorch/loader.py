# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen3-VL Abliterated GGUF model loader implementation for image to text.
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

    The gguf library (>=0.18) knows about qwen3vl tensor names, but
    transformers lacks GGUF loading support for the qwen3vl architecture.
    This bridges the gap, following the same pattern used for gemma3.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

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
        "attention.key_length": "head_dim",
        "vocab_size": "vocab_size",
    }

    # Patch load_gguf_checkpoint: qwen3vl GGUF only has text backbone weights,
    # but Qwen3VLForConditionalGeneration expects a composite config with
    # text_config and vision_config sub-dicts.
    orig_load = gguf_utils.load_gguf_checkpoint

    # Text config keys extracted from GGUF that need to be nested
    _TEXT_CONFIG_KEYS = {
        "max_position_embeddings",
        "num_hidden_layers",
        "intermediate_size",
        "hidden_size",
        "rope_theta",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "head_dim",
        "vocab_size",
    }

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen3vl":
            config = result["config"]
            # Extract text config params from flat GGUF config
            text_config = {"model_type": "qwen3_vl_text"}
            for key in list(config.keys()):
                if key in _TEXT_CONFIG_KEYS:
                    text_config[key] = config.pop(key)
            config["model_type"] = "qwen3_vl"
            config["text_config"] = text_config
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Patch get_gguf_hf_weights_map: map qwen3_vl_text back to qwen3vl
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_vl_text", "qwen3_vl"):
            model_type = "qwen3vl"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # Register tokenizer converter (same as qwen3/qwen2)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available Huihui Qwen3-VL Abliterated GGUF model variants for image to text."""

    HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF = (
        "8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Huihui Qwen3-VL Abliterated GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf",
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
            model="Huihui Qwen3-VL Abliterated GGUF",
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

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
