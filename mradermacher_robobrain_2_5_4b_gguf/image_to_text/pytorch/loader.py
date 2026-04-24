# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher RoboBrain 2.5 4B GGUF model loader implementation for image to text.
"""

import importlib.metadata
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
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


def _patch_qwen3vl_support():
    """Register qwen3vl architecture as an alias for qwen3 in GGUF mappings.

    The RoboBrain2.5 GGUF file declares architecture 'qwen3vl', which
    transformers does not yet recognise. Register it using the same config
    mapping as 'qwen3' and fix model_type to 'qwen3_vl' after loading.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3vl",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support."""
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Mradermacher RoboBrain 2.5 4B GGUF model variants for image to text."""

    ROBOBRAIN_2_5_4B_GGUF = "robobrain_2_5_4b_gguf"


class ModelLoader(ForgeModel):
    """Mradermacher RoboBrain 2.5 4B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ROBOBRAIN_2_5_4B_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/RoboBrain2.5-4B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBOBRAIN_2_5_4B_GGUF

    GGUF_FILE = "RoboBrain2.5-4B.Q4_K_M.gguf"

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mradermacher RoboBrain 2.5 4B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("BAAI/RoboBrain2.5-4B")

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
