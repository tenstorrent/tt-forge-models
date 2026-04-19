# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Felldude Qwen3-VL Uncensored V2 GGUF model loader implementation for image to text.
"""


def _patch_transformers_qwen3vl_gguf():
    """Monkey-patch transformers to add qwen3vl GGUF architecture support."""
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
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    _TEXT_CONFIG_KEYS = {
        "max_position_embeddings",
        "num_hidden_layers",
        "intermediate_size",
        "hidden_size",
        "rope_theta",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "vocab_size",
        "head_dim",
    }

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
            text_config = {}
            for k in list(config.keys()):
                if k in _TEXT_CONFIG_KEYS:
                    text_config[k] = config.pop(k)
            config["text_config"] = text_config
            hidden_size = text_config.get("hidden_size")
            if hidden_size is not None:
                config.setdefault("vision_config", {})["out_hidden_size"] = hidden_size
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen3vl_gguf()

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


class ModelVariant(StrEnum):
    """Available Felldude Qwen3-VL Uncensored V2 GGUF model variants for image to text."""

    QWEN3_VL_8B_INSTRUCT_UNCENSORED_V2_Q4_K_M_GGUF = (
        "8B_INSTRUCT_UNCENSORED_V2_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Felldude Qwen3-VL Uncensored V2 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_8B_INSTRUCT_UNCENSORED_V2_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-8B-Instruct-Uncensored-V2-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_8B_INSTRUCT_UNCENSORED_V2_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN3_VL_8B_INSTRUCT_UNCENSORED_V2_Q4_K_M_GGUF: "Qwen3-VL-8B-Instruct-Uncensored-V2.i1-Q4_K_M.gguf",
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
            model="Felldude Qwen3-VL Uncensored V2 GGUF",
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
