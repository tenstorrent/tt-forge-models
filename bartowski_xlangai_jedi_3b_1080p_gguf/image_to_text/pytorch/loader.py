# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bartowski xlangai Jedi-3B-1080p GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
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


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    The gguf library knows about qwen2vl tensor names, but transformers lacks
    the config mapping and architecture registration needed to load qwen2vl
    GGUF checkpoints.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = {
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

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    _QWEN2VL_TEXT_CONFIG_KEYS = {
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "vocab_size",
    }

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            config["model_type"] = "qwen2_5_vl"
            text_config = {}
            for key in list(config.keys()):
                if key in _QWEN2VL_TEXT_CONFIG_KEYS:
                    text_config[key] = config.pop(key)
            if text_config:
                config["text_config"] = text_config
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

    orig_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen2vl", "qwen2_5_vl"):
            model_type = "qwen2vl"
            if num_layers is None:
                try:
                    tc = hf_model.config.text_config
                    nl = getattr(tc, "num_hidden_layers", None)
                    if nl is not None:
                        num_layers = nl
                except AttributeError:
                    pass
            if num_layers is None:
                try:
                    num_layers = len(hf_model.model.language_model.layers)
                except AttributeError:
                    pass
        return orig_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_qwen2vl_gguf()


class ModelVariant(StrEnum):
    """Available Bartowski xlangai Jedi-3B-1080p GGUF variants for image to text."""

    XLANGAI_JEDI_3B_1080P_GGUF = "xlangai_jedi_3b_1080p_gguf"


class ModelLoader(ForgeModel):
    """Bartowski xlangai Jedi-3B-1080p GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.XLANGAI_JEDI_3B_1080P_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/xlangai_Jedi-3B-1080p-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLANGAI_JEDI_3B_1080P_GGUF

    GGUF_FILE = "xlangai_Jedi-3B-1080p-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Bartowski xlangai Jedi-3B-1080p GGUF",
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
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        # The GGUF only contains quantized text weights; vision config keys are
        # absent from the GGUF metadata, so the default config uses wrong vision
        # dimensions (7B defaults).  Supply the correct 3B config so the vision
        # encoder is instantiated with matching out_hidden_size.
        config = Qwen2_5_VLConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        model_kwargs["config"] = config

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
