# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 32B Thinking GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
)

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
    """Register qwen3vl GGUF architecture as alias for qwen3 config mapping.

    transformers 5.x does not yet recognise the qwen3vl GGUF architecture used
    by Qwen3-VL GGUF checkpoints.  This patches the supported-architectures list
    and the config-field mapping so that load_gguf_checkpoint accepts the file.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3vl",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support and fix model_type."""
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map qwen3_vl HF model type to qwen3vl gguf arch."""
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type == "qwen3_vl":
        model_type = "qwen3vl"
        # Qwen3VLConfig nests num_hidden_layers inside text_config; the base
        # get_gguf_hf_weights_map reads it from the top-level config, so we
        # must resolve it here before handing off.
        if (
            num_layers is None
            and hasattr(hf_model, "config")
            and hasattr(hf_model.config, "text_config")
        ):
            num_layers = hf_model.config.text_config.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 32B Thinking GGUF model variants for image to text."""

    QWEN_3_VL_32B_THINKING_1M_GGUF = "32b_thinking_1m_gguf"
    QWEN_3_VL_32B_THINKING_GGUF = "32b_thinking_gguf"


BASE_MODEL = "Qwen/Qwen3-VL-32B-Thinking"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 32B Thinking GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-32B-Thinking-1M-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_32B_THINKING_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Thinking-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF: "Qwen3-VL-32B-Thinking-1M-Q4_K_M.gguf",
        ModelVariant.QWEN_3_VL_32B_THINKING_GGUF: "Qwen3VL-32B-Thinking-Q4_K_M.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 32B Thinking GGUF",
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
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)

        # Pre-load full VL config: GGUF metadata lacks vision_config needed by
        # Qwen3VLForConditionalGeneration, so pass the base model config explicitly
        # to bypass GGUF config extraction.
        config = AutoConfig.from_pretrained(BASE_MODEL)
        model_kwargs["config"] = config

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
