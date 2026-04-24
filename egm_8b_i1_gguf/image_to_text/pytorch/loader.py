# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EGM-8B i1 GGUF model loader implementation for image to text.
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

    Transformers 5.x has Qwen3VLForConditionalGeneration but lacks GGUF loading
    support for the qwen3vl architecture. We add the config mapping and restructure
    the flat config fields into the nested text_config expected by Qwen3VLConfig.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

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

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    _TEXT_CONFIG_KEYS = [
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
    ]

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
            text_config = {}
            for key in _TEXT_CONFIG_KEYS:
                if key in config:
                    text_config[key] = config.pop(key)
            if text_config:
                config["text_config"] = text_config
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
            if num_layers is None and hasattr(hf_model.config, "text_config"):
                num_layers = hf_model.config.text_config.num_hidden_layers
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available EGM-8B i1 GGUF model variants for image to text."""

    EGM_8B_Q4_K_M_GGUF = "8B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """EGM-8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.EGM_8B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/EGM-8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EGM_8B_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.EGM_8B_Q4_K_M_GGUF: "EGM-8B.i1-Q4_K_M.gguf",
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
            model="EGM-8B i1 GGUF",
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
            "Efficient-Large-Model/EGM-8B",
        )

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
