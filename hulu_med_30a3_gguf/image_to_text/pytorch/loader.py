# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hulu-Med 30A3 GGUF model loader implementation for medical image to text.
"""

from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.integrations.ggml import (
    GGUF_TO_FAST_CONVERTERS,
    GGUF_CONFIG_MAPPING,
    GGUFQwen2Converter,
)


def _patch_qwen3vlmoe_gguf_support():
    """Register qwen3vlmoe GGUF architecture support.

    Hulu-Med-30A3 uses the 'qwen3vlmoe' architecture identifier in its GGUF
    metadata. Transformers has Qwen3VLMoeForConditionalGeneration (model_type
    'qwen3_vl_moe') but lacks GGUF loading support for the qwen3vlmoe
    architecture. The config layout matches qwen3_moe and uses the same
    GGUFQwen2Converter tokenizer.

    gguf-py knows this architecture as 'qwen3vlmoe', so we also patch
    get_gguf_hf_weights_map to remap qwen3_vl_moe -> qwen3vlmoe for tensor
    name lookups.
    """
    if "qwen3vlmoe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    GGUF_CONFIG_MAPPING.setdefault(
        "qwen3vlmoe",
        {
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
        },
    )

    GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vlmoe", GGUFQwen2Converter)

    _prev_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl_moe":
            model_type = "qwen3vlmoe"
        return _prev_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen3vlmoe_gguf_support()

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
    """Available Hulu-Med 30A3 GGUF model variants for image to text."""

    HULU_MED_30A3_GGUF = "30a3_gguf"


class ModelLoader(ForgeModel):
    """Hulu-Med 30A3 GGUF model loader implementation for medical image to text tasks."""

    _VARIANTS = {
        ModelVariant.HULU_MED_30A3_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Hulu-Med-30A3-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HULU_MED_30A3_GGUF

    GGUF_FILE = "Hulu-Med-30A3.Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Hulu-Med 30A3 GGUF",
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
        model_kwargs["low_cpu_mem_usage"] = True
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(
            "ZJU-AI4H/Hulu-Med-30A3", trust_remote_code=True
        )

        # Load config from the base model: the GGUF metadata doesn't carry the
        # nested VL config structure (text_config / vision_config) that
        # Qwen3VLMoeForConditionalGeneration requires, so it maps tensors to
        # wrong shapes. Using the base config ensures correct architecture.
        config = AutoConfig.from_pretrained(
            "ZJU-AI4H/Hulu-Med-30A3", trust_remote_code=True
        )
        model_kwargs["config"] = config

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
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
                    {
                        "type": "text",
                        "text": "Generate a medical report for this image.",
                    },
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
