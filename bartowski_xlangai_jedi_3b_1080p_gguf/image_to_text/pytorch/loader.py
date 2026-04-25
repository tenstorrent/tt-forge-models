# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bartowski xlangai Jedi-3B-1080p GGUF model loader implementation for image to text.
"""

import importlib.metadata

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.utils.import_utils as _import_utils
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
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


def _patch_qwen2vl_support():
    """Register qwen2vl GGUF architecture as an alias for qwen2_5_vl.

    Qwen2.5-VL GGUF files declare architecture 'qwen2vl', which transformers
    does not recognise.  Map its config keys to the qwen2 set (same text
    backbone) and rewrite model_type to 'qwen2_5_vl' after loading.
    """
    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    for section in GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen2" in GGUF_TO_TRANSFORMERS_MAPPING[section]:
            GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen2vl",
                GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2"],
            )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    """Wrap load_gguf_checkpoint to add qwen2vl support and fix model_type."""
    _patch_qwen2vl_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
    )
    if result.get("config", {}).get("model_type") == "qwen2vl":
        result["config"]["model_type"] = "qwen2_5_vl"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor=None, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to handle qwen2_5_vl VL configs.

    Qwen2_5_VLConfig nests num_hidden_layers inside text_config instead of
    exposing it at the top level.  The gguf library names the architecture
    'qwen2vl', while transformers uses 'qwen2_5_vl'.
    """
    if model_type is None or model_type == "qwen2_5_vl":
        model_type = "qwen2vl"
    if num_layers is None:
        cfg = getattr(hf_model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            num_layers = getattr(text_cfg, "num_hidden_layers", None)
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_patch_qwen2vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Bartowski xlangai Jedi-3B-1080p GGUF variants for image to text."""

    XLANGAI_JEDI_3B_1080P_GGUF = "xlangai_jedi_3b_1080p_gguf"


class ModelLoader(ForgeModel):
    """Bartowski xlangai Jedi-3B-1080p GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.XLANGAI_JEDI_3B_1080P_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/xlangai_Jedi-3B-1080p-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLANGAI_JEDI_3B_1080P_GGUF

    GGUF_FILE = "xlangai_Jedi-3B-1080p-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Bartowski xlangai Jedi-3B-1080p GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # Refresh the stale distribution mapping so transformers can resolve the
        # dynamically-installed gguf package version in is_gguf_available().
        _import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

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
