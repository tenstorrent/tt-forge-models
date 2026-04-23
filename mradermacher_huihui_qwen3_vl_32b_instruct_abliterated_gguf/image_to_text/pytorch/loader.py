# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Huihui-Qwen3-VL-32B-Instruct-abliterated GGUF model loader implementation for image to text.
"""

import importlib.metadata

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
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


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if the package was installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


def _patch_qwen3vl_support():
    """Register qwen3vl architecture as an alias for qwen3 in GGUF mappings.

    Qwen3-VL GGUF files declare architecture as 'qwen3vl' which transformers 5.x
    does not yet recognise. The text backbone uses identical config fields to qwen3.
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


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support and remap model_type to qwen3."""
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3"
    return result


_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Patch get_gguf_hf_weights_map to handle Qwen3VLConfig.

    Qwen3VLConfig has num_hidden_layers inside text_config, not at the top level.
    gguf-py uses 'qwen3vl' (no underscore) while transformers uses 'qwen3_vl'.
    This function is called recursively for child modules, so we must not access
    hf_model.config unconditionally (child modules may not have a config attribute).
    """
    if model_type is None:
        config = getattr(hf_model, "config", None)
        model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_vl":
        model_type = "qwen3vl"
        if num_layers is None:
            config = getattr(hf_model, "config", None)
            if config is not None and hasattr(config, "text_config"):
                num_layers = config.text_config.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available mradermacher Huihui-Qwen3-VL-32B-Instruct-abliterated GGUF model variants for image to text."""

    HUIHUI_QWEN3_VL_32B_INSTRUCT_ABLITERATED_GGUF = "32b_instruct_abliterated_gguf"


class ModelLoader(ForgeModel):
    """mradermacher Huihui-Qwen3-VL-32B-Instruct-abliterated GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_VL_32B_INSTRUCT_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3-VL-32B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_VL_32B_INSTRUCT_ABLITERATED_GGUF

    GGUF_FILE = "Huihui-Qwen3-VL-32B-Instruct-abliterated.Q4_K_M.gguf"

    # Non-gated Qwen3-VL model used to obtain the vision config and processor.
    # The vision backbone dimensions are identical across Qwen3-VL sizes.
    _BASE_VL_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mradermacher Huihui-Qwen3-VL-32B-Instruct-abliterated GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_full_config(self):
        """Build a full Qwen3VLConfig from the GGUF text backbone + base vision config."""
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        text_config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        base_config = AutoConfig.from_pretrained(self._BASE_VL_MODEL)
        return Qwen3VLConfig(
            text_config=text_config.to_dict(),
            vision_config=base_config.vision_config.to_dict(),
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.processor = AutoProcessor.from_pretrained(self._BASE_VL_MODEL)

        config = self._build_full_config()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["config"] = config
        model_kwargs |= kwargs

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
