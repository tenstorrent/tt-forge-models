# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi Linear model loader implementation for causal language modeling.
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
    """Available Kimi Linear model variants."""

    KIMI_LINEAR_48B_A3B_INSTRUCT = "48B-A3B-Instruct"


class ModelLoader(ForgeModel):
    """Kimi Linear model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.KIMI_LINEAR_48B_A3B_INSTRUCT: ModelConfig(
            pretrained_model_name="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_LINEAR_48B_A3B_INSTRUCT

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Kimi-Linear",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    @staticmethod
    def _patch_outputrecorder():
        import importlib

        generic = importlib.import_module("transformers.utils.generic")
        if not hasattr(generic, "OutputRecorder"):
            from transformers.utils.output_capturing import OutputRecorder

            generic.OutputRecorder = OutputRecorder

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        self._patch_outputrecorder()

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

        self._patch_cache_get_mask_sizes(model)

        self.config = model.config
        self.model = model
        return model

    @staticmethod
    def _patch_cache_get_mask_sizes(model):
        """Patch KimiLinearCache.get_mask_sizes for transformers >=5.5 compat.

        Transformers 5.5+ passes q_length (int) instead of cache_position (Tensor).
        """
        import sys

        model_module = type(model).__module__
        mod = sys.modules.get(model_module)
        if mod is None:
            return
        cache_cls = getattr(mod, "KimiLinearCache", None)
        if cache_cls is None:
            return

        def patched_get_mask_sizes(self, cache_position, layer_idx):
            if isinstance(cache_position, int):
                query_length = cache_position
            else:
                query_length = cache_position.shape[0]
            kv_offset = 0
            past_seen_tokens = self.get_seq_length(layer_idx)
            kv_length = query_length + past_seen_tokens
            return kv_length, kv_offset

        cache_cls.get_mask_sizes = patched_get_mask_sizes

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        test_input = "What is the capital of France?"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
