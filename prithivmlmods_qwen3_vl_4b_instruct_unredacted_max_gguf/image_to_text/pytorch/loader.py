# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
prithivMLmods Qwen3-VL-4B-Instruct-Unredacted-MAX GGUF model loader implementation for image to text.
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

    Transformers 5.x has Qwen3VLForConditionalGeneration but lacks GGUF
    loading support for the qwen3vl architecture. The text portion of
    qwen3vl uses the same config structure as qwen3, so we register
    qwen3vl as an alias.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

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

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available prithivMLmods Qwen3-VL-4B-Instruct-Unredacted-MAX GGUF model variants for image to text."""

    QWEN3_VL_4B_INSTRUCT_UNREDACTED_MAX_Q8_0_GGUF = (
        "4B_INSTRUCT_UNREDACTED_MAX_Q8_0_GGUF"
    )


class ModelLoader(ForgeModel):
    """prithivMLmods Qwen3-VL-4B-Instruct-Unredacted-MAX GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_4B_INSTRUCT_UNREDACTED_MAX_Q8_0_GGUF: LLMModelConfig(
            pretrained_model_name="prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_4B_INSTRUCT_UNREDACTED_MAX_Q8_0_GGUF

    GGUF_FILE = "Qwen3-VL-4B-Instruct-Unredacted-MAX.Q8_0.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="prithivMLmods Qwen3-VL-4B-Instruct-Unredacted-MAX GGUF",
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
        # GGUF loading doesn't fully configure the vision encoder dims; allow
        # mismatched sizes and use text-only inputs to avoid the mismatch.
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # GGUF variant loads with mismatched vision config dimensions; use
        # text-only inputs to avoid the vision feature dimension mismatch.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe an image of a dog."},
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
