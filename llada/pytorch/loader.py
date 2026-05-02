# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaDA (Large Language Diffusion with mAsking) model loader implementation.
"""
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available LLaDA model variants."""

    LLADA_8B_BASE = "8B_Base"


class ModelLoader(ForgeModel):
    """LLaDA model loader implementation for masked diffusion language modeling."""

    _VARIANTS = {
        ModelVariant.LLADA_8B_BASE: LLMModelConfig(
            pretrained_model_name="GSAI-ML/LLaDA-8B-Base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLADA_8B_BASE

    sample_text = "The quick brown fox jumps over the lazy dog."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LLaDA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # LLaDAModelLM.__init__ predates transformers 5.x:
        # (a) post_init() is never called, so all_tied_weights_keys is never set;
        # (b) tie_weights() doesn't accept the missing_keys/recompute_mapping kwargs
        #     that _finalize_model_loading now passes.
        # Patch _finalize_model_loading to fix both before delegating to the original.
        import inspect

        _orig_finalize = PreTrainedModel.__dict__["_finalize_model_loading"].__func__

        @staticmethod
        def _patched_finalize(model, load_config, loading_info):
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(
                    all_submodels=False
                )
            _model_class = type(model)
            _orig_tie = _model_class.__dict__.get("tie_weights")
            if _orig_tie is not None:
                params = inspect.signature(_orig_tie).parameters
                if "recompute_mapping" not in params and "kwargs" not in params:

                    def _compat_tie(self, **kwargs):
                        return _orig_tie(self)

                    _model_class.tie_weights = _compat_tie
            try:
                return _orig_finalize(model, load_config, loading_info)
            finally:
                if _orig_tie is not None and _model_class.__dict__.get("tie_weights") is not _orig_tie:
                    _model_class.tie_weights = _orig_tie

        PreTrainedModel._finalize_model_loading = _patched_finalize
        try:
            model = AutoModel.from_pretrained(
                pretrained_model_name, trust_remote_code=True, **model_kwargs
            )
        finally:
            PreTrainedModel._finalize_model_loading = staticmethod(_orig_finalize)
        # transformers 5.x pops use_cache from config kwargs (it's a generation
        # parameter); the remote LLaDAModelLM.forward still reads config.use_cache.
        model.config.use_cache = False
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
