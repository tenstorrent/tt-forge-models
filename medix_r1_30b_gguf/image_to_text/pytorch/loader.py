# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 30B GGUF model loader implementation for image to text.
"""

from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
from transformers.integrations.ggml import GGUF_CONFIG_MAPPING, GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_qwen3vlmoe_support():
    """Register qwen3vlmoe architecture so load_gguf_checkpoint accepts it.

    Qwen3 VL MoE GGUF files declare architecture as 'qwen3vlmoe', which
    transformers does not recognise. Register the same config key mappings
    as qwen3_moe so the checkpoint can be parsed, then fix model_type to
    'qwen3_vl_moe' afterwards.
    """
    if "qwen3vlmoe" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["qwen3vlmoe"] = GGUF_CONFIG_MAPPING["qwen3_moe"]
    if "qwen3vlmoe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")
    if "qwen3vlmoe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vlmoe"] = GGUF_TO_FAST_CONVERTERS.get(
            "qwen3_moe", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vlmoe → qwen3_vl_moe support."""
    _patch_qwen3vlmoe_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen3vlmoe":
        result["config"]["model_type"] = "qwen3_vl_moe"
    return result


_patch_qwen3vlmoe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available MediX R1 30B GGUF model variants for image to text."""

    MEDIX_R1_30B_Q4_K_M = "30b_q4_k_m"


class ModelLoader(ForgeModel):
    """MediX R1 30B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEDIX_R1_30B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="MBZUAI/MediX-R1-30B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.MEDIX_R1_30B_Q4_K_M: "MediX-R1-30B-Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.MEDIX_R1_30B_Q4_K_M

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MediX R1 30B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import importlib.metadata
        import transformers.utils.import_utils as _triu

        # Refresh transformers' package distribution cache so dynamically-installed
        # gguf is found. transformers 5.x caches this mapping at import time, before
        # gguf is installed by RequirementsManager.
        _triu.PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = gguf_file
        model_kwargs |= kwargs

        from transformers import Qwen3VLMoeConfig

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        )

        # Load config from base model so that the nested vision_config/text_config
        # are correctly populated. load_gguf_checkpoint returns a flat config dict
        # that does not map to Qwen3VLMoeConfig's nested structure, causing the
        # vision projector out_hidden_size to mismatch the text hidden_size.
        config = Qwen3VLMoeConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
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
