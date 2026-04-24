# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Olmo3 Recurrent Adapter SFT CoT causal LM model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel
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


class ModelVariant(StrEnum):
    """Available Olmo3 Recurrent Adapter SFT CoT model variants for causal language modeling."""

    Olmo3_Recurrent_Adapter_SFT_CoT = "recurrent_adapter_sft_cot"


class ModelLoader(ForgeModel):
    """Olmo3 Recurrent Adapter SFT CoT model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.Olmo3_Recurrent_Adapter_SFT_CoT: LLMModelConfig(
            pretrained_model_name="hanseungwook/olmo3-recurrent-adapter-sft-cot",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Olmo3_Recurrent_Adapter_SFT_CoT

    # The recurrent adapter reuses the OLMo-3-1025-7B base tokenizer.
    tokenizer_model_name = "allenai/Olmo-3-1025-7B"

    sample_text = "What is 25 * 37?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="olmo3_recurrent_adapter_sft_cot",
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
            self.tokenizer_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        # The custom model's __init__ calls AutoModelForCausalLM.from_pretrained for a
        # sub-model; initialize_weights() then runs inside the outer meta device context
        # (transformers 5.x). Non-persistent buffers like inv_freq stay meta, and
        # rope_fn creates more meta tensors, causing "Cannot copy from meta tensor".
        # Also, RecurrentAdapterModel.__init__ never calls self.post_init(), so
        # all_tied_weights_keys is unset when _finalize_model_loading runs.
        # Fix 1: patch get_init_context to skip meta so nested from_pretrained works.
        # Fix 2: patch __init__ to call post_init() so all_tied_weights_keys is set.
        _tmp_cfg = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        _class_ref = getattr(_tmp_cfg, "auto_map", {}).get("AutoModelForCausalLM")
        _model_cls = None
        _orig_gic = None
        _orig_init = None
        if _class_ref:
            _model_cls = get_class_from_dynamic_module(
                _class_ref, pretrained_model_name
            )
            _orig_gic = _model_cls.__dict__.get("get_init_context")
            _orig_init = _model_cls.__dict__.get("__init__")

            @classmethod
            def _no_meta_gic(cls, dtype, is_quantized, _is_ds_init_called):
                ctxs = _PreTrainedModel.get_init_context.__func__(
                    cls, dtype, is_quantized, _is_ds_init_called
                )
                return [
                    c
                    for c in ctxs
                    if not (isinstance(c, torch.device) and c.type == "meta")
                ]

            def _init_with_post_init(self, config):
                _orig_init(self, config)
                if not hasattr(self, "all_tied_weights_keys"):
                    self.post_init()

            _model_cls.get_init_context = _no_meta_gic
            _model_cls.__init__ = _init_with_post_init

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            if _model_cls is not None:
                if _orig_gic is None:
                    del _model_cls.get_init_context
                else:
                    _model_cls.get_init_context = _orig_gic
                if _orig_init is None:
                    del _model_cls.__init__
                else:
                    _model_cls.__init__ = _orig_init
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.config
