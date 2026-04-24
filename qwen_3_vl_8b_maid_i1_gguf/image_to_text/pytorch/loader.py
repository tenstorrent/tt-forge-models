# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 8B Maid i1 GGUF model loader implementation for image to text.
"""

import importlib.metadata

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from typing import Optional


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if the package was installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


def _patch_qwen3vl_support():
    """Register qwen3vl architecture as alias for qwen3.

    The GGUF file declares architecture as 'qwen3vl', which transformers does
    not yet recognise. Map it to the qwen3 backbone entries.
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


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    """Wrap load_gguf_checkpoint to add qwen3vl support and fix model_type."""
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


def _apply_patches():
    """Apply all patches. Called at import time and again just before loading,
    because other model loaders imported later by pytest can overwrite the patches."""
    _patch_qwen3vl_support()
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_apply_patches()

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
    """Available Qwen 3 VL 8B Maid i1 GGUF model variants for image to text."""

    QWEN_3_VL_8B_MAID_I1_GGUF = "8b_maid_i1_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 8B Maid i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_8B_MAID_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-8B-maid-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_8B_MAID_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_VL_8B_MAID_I1_GGUF: "Qwen3-VL-8B-maid.i1-Q4_K_M.gguf",
    }

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
            model="Qwen 3 VL 8B Maid i1 GGUF",
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
        _refresh_gguf_detection()
        _apply_patches()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

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
