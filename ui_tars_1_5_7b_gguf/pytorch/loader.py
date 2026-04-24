# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UI-TARS 1.5 7B GGUF model loader implementation for vision-language GUI agent tasks.

The GGUF checkpoint declares architecture ``qwen2vl``, which transformers
does not recognise for GGUF. We alias it to ``qwen2`` during GGUF loading
and rewrite the resulting model_type to ``qwen2_5_vl_text`` so it slots into
the full ``Qwen2_5_VLConfig`` (the base UI-TARS 1.5 model is a Qwen2.5-VL
variant; its vision backbone is pulled from the corresponding non-GGUF base
model).

Repository:
- https://huggingface.co/Mungert/UI-TARS-1.5-7B-GGUF
"""
import importlib.metadata

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from typing import Optional


from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


def _patch_qwen2vl_support():
    """Register ``qwen2vl`` as an alias of ``qwen2`` for GGUF loading."""
    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]
        if "qwen2" in mapping:
            mapping["qwen2vl"] = mapping["qwen2"]
    if "qwen2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUF_TO_FAST_CONVERTERS["qwen2"]


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to accept qwen2vl and relabel its model_type."""
    _patch_qwen2vl_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen2vl":
        result["config"]["model_type"] = "qwen2_5_vl_text"
    return result


_patch_qwen2vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


class ModelVariant(StrEnum):
    """Available UI-TARS 1.5 7B GGUF model variants."""

    Q4_K = "Q4_K"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """UI-TARS 1.5 7B GGUF model loader implementation for vision-language GUI agent tasks."""

    _VARIANTS = {
        ModelVariant.Q4_K: ModelConfig(
            pretrained_model_name="Mungert/UI-TARS-1.5-7B-GGUF",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="Mungert/UI-TARS-1.5-7B-GGUF",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K: "UI-TARS-1.5-7B-q4_k_m.gguf",
        ModelVariant.Q8_0: "UI-TARS-1.5-7B-q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    # Processor / base non-GGUF model (source of vision_config and processor).
    _BASE_MODEL = "ByteDance-Seed/UI-TARS-1.5-7B"

    # Shared configuration parameters — text-only prompt. The Qwen2.5-VL
    # vision encoder uses data-dependent Python control flow that TT-XLA
    # cannot compile; feed text-only inputs so the vision branch is skipped.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe an image."},
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="UI-TARS 1.5 7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor from the original non-GGUF model."""
        self.processor = AutoProcessor.from_pretrained(self._BASE_MODEL)
        return self.processor

    def _build_full_config(self):
        """Combine the text config pulled from GGUF with the base model's vision config."""
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]
        text_config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )
        base_config = AutoConfig.from_pretrained(self._BASE_MODEL)
        # The GGUF loader produces a plain qwen2-style rope_parameters; the
        # Qwen2.5-VL attention layer requires the mrope_section that only the
        # base (non-GGUF) config carries.
        text_config.rope_parameters = dict(base_config.text_config.rope_parameters)
        return Qwen2_5_VLConfig(
            text_config=text_config.to_dict(),
            vision_config=base_config.vision_config.to_dict(),
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_file
        model_kwargs["config"] = self._build_full_config()

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
