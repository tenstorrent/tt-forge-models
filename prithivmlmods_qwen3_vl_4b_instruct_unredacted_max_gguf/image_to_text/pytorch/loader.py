# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
prithivMLmods Qwen3-VL-4B-Instruct-Unredacted-MAX GGUF model loader implementation for image to text.
"""

import importlib.metadata

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from typing import Optional

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


def _patch_qwen3vl_gguf_support():
    """Register qwen3vl GGUF architecture support in transformers.

    transformers does not yet natively support the qwen3vl GGUF architecture.
    This patches three things:
    1. Adds qwen3vl to GGUF_SUPPORTED_ARCHITECTURES to bypass the check.
    2. Adds config field mapping (same keys as qwen3 text backbone).
    3. Patches load_gguf_checkpoint to remap model_type qwen3vl -> qwen3_vl.
    4. Patches get_gguf_hf_weights_map to resolve the qwen3_vl model type to the
       qwen3vl GGUF architecture name used by the gguf-py library.
    5. Fixes the PACKAGE_DISTRIBUTION_MAPPING stale-cache issue for runtime gguf installs.
    """
    import transformers.utils.import_utils as _import_utils

    # Fix gguf version detection when installed at runtime by RequirementsManager.
    if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        try:
            importlib.metadata.version("gguf")
            _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
            _import_utils.is_gguf_available.cache_clear()
        except importlib.metadata.PackageNotFoundError:
            pass

    if "qwen3vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    if "qwen3vl" not in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]:
        qwen3_cfg = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].get("qwen3", {})
        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = dict(qwen3_cfg)

    # Patch load_gguf_checkpoint to fix model_type after parsing.
    def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
        result = _orig_load_gguf_checkpoint(
            gguf_path, return_tensors=return_tensors, **kwargs
        )
        if result.get("config", {}).get("model_type") == "qwen3vl":
            result["config"]["model_type"] = "qwen3_vl"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    import transformers.configuration_utils as _config_utils

    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Patch get_gguf_hf_weights_map to recognise qwen3_vl / qwen3_vl_text as qwen3vl.
    _orig_map_fn = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, **kwargs
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_vl", "qwen3_vl_text"):
            model_type = "qwen3vl"
        return _orig_map_fn(hf_model, processor, model_type=model_type, **kwargs)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen3vl_gguf_support()


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

    _BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

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
        model_kwargs |= kwargs

        # The GGUF repo has no config.json; load config from the base model so
        # AutoConfig.from_pretrained doesn't try to parse qwen3vl from the GGUF.
        model_kwargs.setdefault("config", AutoConfig.from_pretrained(self._BASE_MODEL))

        self.processor = AutoProcessor.from_pretrained(self._BASE_MODEL)

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
