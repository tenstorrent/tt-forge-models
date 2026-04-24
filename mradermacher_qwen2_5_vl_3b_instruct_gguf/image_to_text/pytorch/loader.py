# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher Qwen2.5-VL 3B Instruct GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from typing import Optional

# Qwen2_5_VLConfig nests num_hidden_layers under text_config; expose it at
# the top level so get_gguf_hf_weights_map can find it during GGUF weight mapping.
if not hasattr(Qwen2_5_VLConfig, "num_hidden_layers"):
    Qwen2_5_VLConfig.num_hidden_layers = property(
        lambda self: self.text_config.num_hidden_layers
    )

_QWEN2VL_CONFIG_MAPPING = {
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


def _patch_qwen2vl_support():
    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "qwen2vl", _QWEN2VL_CONFIG_MAPPING
    )
    GGUF_TO_FAST_CONVERTERS.setdefault("qwen2vl", GGUFQwen2Converter)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    _patch_qwen2vl_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen2vl":
        result["config"]["model_type"] = "qwen2_5_vl"
    return result


_patch_qwen2vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
    """Available Mradermacher Qwen2.5-VL 3B Instruct GGUF variants for image to text."""

    QWEN2_5_VL_3B_INSTRUCT_Q4_K_M_GGUF = "3B_INSTRUCT_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mradermacher Qwen2.5-VL 3B Instruct GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN2_5_VL_3B_INSTRUCT_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen2.5-VL-3B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN2_5_VL_3B_INSTRUCT_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN2_5_VL_3B_INSTRUCT_Q4_K_M_GGUF: "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf",
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
            model="Mradermacher Qwen2.5-VL 3B Instruct GGUF",
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
            "Qwen/Qwen2.5-VL-3B-Instruct",
        )

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
