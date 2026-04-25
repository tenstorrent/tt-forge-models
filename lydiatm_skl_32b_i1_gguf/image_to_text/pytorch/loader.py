# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/LydiaTM-SKL-32B-i1-GGUF model loader implementation for image to text.
"""


def _patch_transformers_qwen3vl_gguf():
    """Add qwen3vl GGUF architecture support so Qwen3-VL GGUF files load without error.

    The qwen3vl GGUF architecture is not registered in transformers. Both issues
    are patched here by registering the architecture and its config/converter mappings.
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

from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
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
    """Available mradermacher/LydiaTM-SKL-32B-i1-GGUF model variants for image to text."""

    LYDIATM_SKL_32B_I1_GGUF = "LydiaTM_SKL_32B_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/LydiaTM-SKL-32B-i1-GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.LYDIATM_SKL_32B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/LydiaTM-SKL-32B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LYDIATM_SKL_32B_I1_GGUF

    GGUF_FILE = "LydiaTM-SKL-32B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher LydiaTM-SKL-32B i1 GGUF",
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
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")

        # Provide base config so from_pretrained skips config-from-GGUF:
        # qwen3vl is not in the VL config mapping so the architecture check fails
        # without the patch above; passing a ready-made Qwen3VLConfig avoids that
        # path entirely and ensures vision/text sub-configs are properly populated.
        config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
        # Qwen3VLConfig is a composite config; get_gguf_hf_weights_map reads
        # config.num_hidden_layers directly, so expose it at the top level.
        config.num_hidden_layers = config.text_config.num_hidden_layers
        model_kwargs["config"] = config

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
