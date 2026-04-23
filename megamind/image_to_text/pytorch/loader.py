# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Megamind VL model loader implementation for image to text.
"""

from transformers import (
    AutoConfig,
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
    """Available Megamind VL model variants for image to text."""

    MEGAMIND_V2_VL_HIGH = "v2_vl_high"
    MEGAMIND_V2_VL_HIGH_GGUF = "v2_vl_high_gguf"


class ModelLoader(ForgeModel):
    """Megamind VL model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEGAMIND_V2_VL_HIGH: LLMModelConfig(
            pretrained_model_name="digitranslab/Megamind-v2-VL-high",
            max_length=128,
        ),
        ModelVariant.MEGAMIND_V2_VL_HIGH_GGUF: LLMModelConfig(
            pretrained_model_name="digitranslab/Megamind-v2-VL-high-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEGAMIND_V2_VL_HIGH

    _GGUF_FILES = {
        ModelVariant.MEGAMIND_V2_VL_HIGH_GGUF: "Megamind-v2-VL-high-Q4_K_M.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Megamind VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_gguf_variant(self):
        return self._variant in self._GGUF_FILES

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = "auto"
            model_kwargs["device_map"] = "auto"

        if self._is_gguf_variant():
            model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]
            model_kwargs["ignore_mismatched_sizes"] = True

        model_kwargs |= kwargs

        if self._is_gguf_variant():
            # GGUF repos do not ship a processor; use the base model
            self.processor = AutoProcessor.from_pretrained(
                "digitranslab/Megamind-v2-VL-high"
            )
            # Load full config from the base model so the vision config
            # is correct (GGUF metadata only stores text model fields).
            model_kwargs["config"] = AutoConfig.from_pretrained(
                "digitranslab/Megamind-v2-VL-high"
            )
        else:
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

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
