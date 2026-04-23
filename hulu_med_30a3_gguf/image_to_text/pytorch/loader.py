# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hulu-Med 30A3 GGUF model loader implementation for medical image to text.
"""

from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional


def _patch_transformers_qwen3vlmoe_gguf():
    """Monkey-patch transformers to add GGUF support for the qwen3vlmoe architecture.

    Transformers 5.x has Qwen3VLMoeForConditionalGeneration but lacks GGUF loading
    support for the 'qwen3vlmoe' architecture identifier used in GGUF metadata.
    We bridge the gap by registering the config and tensor processor mappings.
    The gguf-py library already defines MODEL_ARCH.QWEN3VLMOE so tensor name
    resolution works once the architecture is registered.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_CONFIG_MAPPING,
        TENSOR_PROCESSORS,
        Qwen2MoeTensorProcessor,
    )
    from transformers.integrations.ggml import GGUF_CONFIG_DEFAULTS_MAPPING

    if "qwen3vlmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # Text config field mappings mirror qwen3_moe (the LM backbone is identical)
    GGUF_CONFIG_MAPPING["qwen3vlmoe"] = dict(GGUF_CONFIG_MAPPING["qwen3_moe"])
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    # norm_topk_prob must be True for llama.cpp compatibility (same as qwen3_moe)
    GGUF_CONFIG_DEFAULTS_MAPPING["qwen3vlmoe"] = dict(
        GGUF_CONFIG_DEFAULTS_MAPPING.get("qwen3_moe", {})
    )

    # Qwen2MoeTensorProcessor handles the MoE expert weight merging for text layers
    TENSOR_PROCESSORS["qwen3vlmoe"] = Qwen2MoeTensorProcessor


_patch_transformers_qwen3vlmoe_gguf()

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
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(
            "ZJU-AI4H/Hulu-Med-30A3", trust_remote_code=True
        )

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
