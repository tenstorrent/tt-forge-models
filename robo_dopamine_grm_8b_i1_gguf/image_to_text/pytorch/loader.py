# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Robo-Dopamine GRM 8B i1 GGUF model loader implementation for image to text.
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
    Qwen2_5_VLForConditionalGeneration,
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


def _patch_qwen2vl_gguf_support():
    """Register qwen2vl GGUF architecture as an alias for qwen2_5_vl.

    Qwen2-VL GGUF files declare architecture as 'qwen2vl' but transformers
    only recognises 'qwen2_5_vl' as a model type and has no GGUF config
    mapping for 'qwen2vl'. The text backbone uses the same tensor layout as
    qwen2.
    """
    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    config_section = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]
    if "qwen2vl" not in config_section and "qwen2" in config_section:
        config_section["qwen2vl"] = config_section["qwen2"]
    if "qwen2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen2vl", GGUF_TO_FAST_CONVERTERS["qwen2"])


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen2vl support and fix model_type."""
    _patch_qwen2vl_gguf_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen2vl":
        result["config"]["model_type"] = "qwen2_5_vl"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to map qwen2_5_vl to gguf-py's qwen2vl arch."""
    effective_type = hf_model.config.model_type if model_type is None else model_type
    if effective_type == "qwen2_5_vl":
        model_type = "qwen2vl"
        if num_layers is None:
            num_layers = hf_model.config.text_config.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_patch_qwen2vl_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Robo-Dopamine GRM 8B i1 GGUF model variants for image to text."""

    ROBO_DOPAMINE_GRM_8B_I1_GGUF = "8b_i1_gguf"


class ModelLoader(ForgeModel):
    """Robo-Dopamine GRM 8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ROBO_DOPAMINE_GRM_8B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Robo-Dopamine-GRM-8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBO_DOPAMINE_GRM_8B_I1_GGUF

    GGUF_FILE = "Robo-Dopamine-GRM-8B.i1-Q4_K_M.gguf"

    # Processor and full config source (the GGUF repo ships only quantized weights).
    _BASE_MODEL_SOURCE = "tanhuajie2001/Robo-Dopamine-GRM-8B"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Robo-Dopamine GRM 8B i1 GGUF",
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

        self.processor = AutoProcessor.from_pretrained(self._BASE_MODEL_SOURCE)

        # Load the full config from the base model so vision_config is populated
        # (GGUF parsing only sets text model params).
        config = AutoConfig.from_pretrained(self._BASE_MODEL_SOURCE)
        model_kwargs["config"] = config

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        ).eval()

        self.config = model.config
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self._BASE_MODEL_SOURCE)
        return self.config
