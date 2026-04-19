# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InteractiveOmni-4B model loader implementation for multimodal conditional generation.
"""

import sys
import types
from typing import Optional

from transformers import AutoConfig, AutoModel, AutoTokenizer

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available InteractiveOmni model variants."""

    INTERACTIVE_OMNI_4B = "4B"


class ModelLoader(ForgeModel):
    """InteractiveOmni model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.INTERACTIVE_OMNI_4B: ModelConfig(
            pretrained_model_name="sensenova/InteractiveOmni-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERACTIVE_OMNI_4B

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize InteractiveOmni model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InteractiveOmni",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        return self.tokenizer

    @staticmethod
    def _patch_transformers_onnx():
        import transformers

        _onnx_stub = types.ModuleType("transformers.onnx")
        _onnx_stub.OnnxConfig = type("OnnxConfig", (), {})
        _onnx_stub.OnnxSeq2SeqConfigWithPast = type("OnnxSeq2SeqConfigWithPast", (), {})
        sys.modules["transformers.onnx"] = _onnx_stub
        transformers.onnx = _onnx_stub

    @staticmethod
    def _force_eager_attn(config):
        config._attn_implementation_internal = "eager"
        config._attn_implementation_autoset = False
        for attr in vars(config):
            child = getattr(config, attr)
            if hasattr(child, "_attn_implementation_internal"):
                child._attn_implementation_internal = "eager"
                child._attn_implementation_autoset = False

    @staticmethod
    def _get_model_class(model_name):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(str(model_name), trust_remote_code=True)
        class_ref = config.auto_map.get("AutoModel")
        if class_ref:
            return get_class_from_dynamic_module(
                class_ref, model_name, trust_remote_code=True
            )
        return None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the InteractiveOmni model instance."""
        self._patch_transformers_onnx()
        model_name = self._variant_config.pretrained_model_name

        config = AutoConfig.from_pretrained(str(model_name), trust_remote_code=True)
        self._force_eager_attn(config)

        # The model's custom from_pretrained loads an ONNX speaker encoder
        # and runs audio feature extraction that crashes with random weights.
        # Temporarily remove it so the base from_pretrained is used instead.
        model_class = self._get_model_class(model_name)
        orig_from_pretrained = None
        if model_class and "from_pretrained" in model_class.__dict__:
            orig_from_pretrained = model_class.__dict__["from_pretrained"]
            delattr(model_class, "from_pretrained")

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "config": config,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        try:
            model = AutoModel.from_pretrained(str(model_name), **model_kwargs)
        finally:
            if orig_from_pretrained is not None:
                model_class.from_pretrained = orig_from_pretrained

        # The top-level model has no forward(); extract the LLM backbone
        # which accepts standard input_ids / attention_mask for compilation.
        model = model.language_model
        model.eval()

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for InteractiveOmni."""
        if self.tokenizer is None:
            self._load_tokenizer()

        # Build prompt
        messages = [{"role": "user", "content": self.sample_text}]
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            text_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
