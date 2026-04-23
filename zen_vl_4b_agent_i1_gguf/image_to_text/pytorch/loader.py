# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zen VL 4B Agent i1 GGUF model loader implementation for image to text.
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

    Transformers 5.x supports Qwen3VLForConditionalGeneration but lacks GGUF
    loading support for the qwen3vl architecture. The text LLM component uses
    the same key layout as qwen3, so we bridge the config mapping and restructure
    the flat GGUF config into Qwen3VL's nested text_config/vision_config format.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3vl as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    # 2. Map qwen3vl GGUF keys the same way as qwen3 (text LLM portion)
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

    # 3. Register BPE tokenizer converter (qwen3vl uses the same BPE scheme as qwen3)
    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    # 4. Patch load_gguf_checkpoint to restructure flat config into Qwen3VL nested format
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            # Move text LLM keys into a nested text_config dict so that
            # Qwen3VLConfig can be constructed with the correct dimensions.
            text_keys = {
                "max_position_embeddings",
                "num_hidden_layers",
                "intermediate_size",
                "hidden_size",
                "rope_theta",
                "num_attention_heads",
                "num_key_value_heads",
                "rms_norm_eps",
                "vocab_size",
                "tie_word_embeddings",
            }
            text_config = {
                k: config.pop(k) for k in list(config.keys()) if k in text_keys
            }
            config["text_config"] = text_config
            # Transformers uses "qwen3_vl" (with underscore), not "qwen3vl"
            config["model_type"] = "qwen3_vl"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Also update modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


# Apply patch at import time so all downstream from_pretrained calls benefit
_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available Zen VL 4B Agent i1 GGUF model variants for image to text."""

    ZEN_VL_4B_AGENT_I1_GGUF = "4b_agent_i1_gguf"


class ModelLoader(ForgeModel):
    """Zen VL 4B Agent i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ZEN_VL_4B_AGENT_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/zen-vl-4b-agent-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ZEN_VL_4B_AGENT_I1_GGUF

    GGUF_FILE = "zen-vl-4b-agent.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Zen VL 4B Agent i1 GGUF",
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
        self.processor = AutoProcessor.from_pretrained("zenlm/zen-vl-4b-agent")

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
