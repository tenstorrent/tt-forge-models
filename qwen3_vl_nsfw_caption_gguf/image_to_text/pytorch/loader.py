# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 VL NSFW Caption GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)

from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
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
    """Register qwen3vl GGUF architecture as an alias for qwen3_vl.

    Qwen3-VL GGUF files declare architecture as 'qwen3vl' but transformers 5.x
    only recognises 'qwen3_vl' as a model type and has no GGUF config mapping
    for 'qwen3vl'. The text backbone uses the same tensor layout as qwen3.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    config_section = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]
    if "qwen3vl" not in config_section and "qwen3" in config_section:
        config_section["qwen3vl"] = config_section["qwen3"]
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support and fix model_type."""
    _patch_qwen3vl_gguf_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map qwen3_vl to gguf-py's qwen3vl arch."""
    effective_type = hf_model.config.model_type if model_type is None else model_type
    if effective_type == "qwen3_vl":
        model_type = "qwen3vl"
        # Qwen3VLConfig nests num_hidden_layers under text_config; provide it
        # explicitly so the original function doesn't call hf_model.config.num_hidden_layers
        if num_layers is None:
            num_layers = hf_model.config.text_config.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_patch_qwen3vl_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Qwen3 VL NSFW Caption GGUF model variants for image to text."""

    QWEN3_VL_8B_NSFW_CAPTION_V4_5_GGUF = "8b_nsfw_caption_v4_5_gguf"
    QWEN3_VL_8B_NSFW_CAPTION_V4_5_HERETIC_I1_GGUF = (
        "8b_nsfw_caption_v4_5_heretic_i1_gguf"
    )


class ModelLoader(ForgeModel):
    """Qwen3 VL NSFW Caption GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_8B_NSFW_CAPTION_V4_5_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN3_VL_8B_NSFW_CAPTION_V4_5_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-8B-NSFW-Caption-V4.5-heretic-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_8B_NSFW_CAPTION_V4_5_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN3_VL_8B_NSFW_CAPTION_V4_5_GGUF: "Qwen3-VL-8B-NSFW-Caption-V4.5.Q4_K_M.gguf",
        ModelVariant.QWEN3_VL_8B_NSFW_CAPTION_V4_5_HERETIC_I1_GGUF: "Qwen3-VL-8B-NSFW-Caption-V4.5-heretic.i1-Q4_K_M.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen3 VL NSFW Caption GGUF",
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
        model_kwargs["gguf_file"] = self.gguf_file
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor or vision config; use the official base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        # Load the full config from official base model so vision_config.out_hidden_size
        # matches the text model's hidden_size (GGUF parsing only sets text model params)
        config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
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
