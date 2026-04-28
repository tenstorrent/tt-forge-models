# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MPT-1B RedPajama 200B Dolly causal language model loader implementation.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MPT-1B RedPajama 200B Dolly model variants for causal language modeling."""

    MPT_1B_REDPAJAMA_200B_DOLLY = "mpt-1b-redpajama-200b-dolly"


class ModelLoader(ForgeModel):
    """MPT-1B RedPajama 200B Dolly model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MPT_1B_REDPAJAMA_200B_DOLLY: ModelConfig(
            pretrained_model_name="anas-awadalla/mpt-1b-redpajama-200b-dolly",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MPT_1B_REDPAJAMA_200B_DOLLY

    sample_text = "Hey how are you doing today?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MPT-1B-RedPajama-200B-Dolly",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    @staticmethod
    def _patch_mosaic_gpt(pretrained_model_name):
        # MosaicGPT was written before transformers 5.x made post_init() mandatory.
        # Its __init__ never calls self.post_init(), so from_pretrained fails in
        # _finalize_model_loading when it accesses self.all_tied_weights_keys.
        # Also patch _attn_bias to reinitialize when the device changes: the test
        # framework runs a CPU forward for the golden reference first, which sets
        # self.attn_bias on CPU; then when compiling for XLA the cached CPU tensor
        # conflicts with XLA fake tensors causing a device mismatch error.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.modeling_utils import PreTrainedModel

        try:
            mosaic_cls = get_class_from_dynamic_module(
                "mosaic_gpt.MosaicGPT", pretrained_model_name
            )
        except Exception:
            return

        if getattr(mosaic_cls, "_tt_patched", False):
            return

        # Patch 1: call post_init() if __init__ didn't
        _orig_init = mosaic_cls.__init__

        def _patched_init(self, config, _orig=_orig_init):
            _orig(self, config)
            if not hasattr(self, "all_tied_weights_keys"):
                PreTrainedModel.post_init(self)

        mosaic_cls.__init__ = _patched_init

        # Patch 2: reset attn_bias when device changes so XLA gets a fresh bias
        _orig_attn_bias = mosaic_cls._attn_bias

        def _device_aware_attn_bias(self, device, dtype, *args, _orig=_orig_attn_bias, **kwargs):
            if self._attn_bias_initialized and self.attn_bias is not None:
                try:
                    if self.attn_bias.device.type != torch.device(device).type:
                        self._attn_bias_initialized = False
                        self.attn_bias = None
                except Exception:
                    pass
            return _orig(self, device, dtype, *args, **kwargs)

        mosaic_cls._attn_bias = _device_aware_attn_bias
        mosaic_cls._tt_patched = True

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        self._patch_mosaic_gpt(pretrained_model_name)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
