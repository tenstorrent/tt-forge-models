# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron-H model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def _patch_model_for_cpu(model):
    """Apply patches to run Nemotron-H on non-CUDA devices.

    1. Replace torch.cuda.stream in NemotronHBlock.forward with plain code.
    2. Stub _update_causal_mask to return None, avoiding a Dynamo shape
       mismatch during tracing. The attention blocks already fall back to
       is_causal=True when no mask is provided.
    """
    for module in model.modules():
        if type(module).__name__ == "NemotronHBlock":
            block_cls = type(module)
            break
    else:
        return

    def patched_forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(
                hidden_states,
                cache_params=kwargs.get("cache_params"),
                cache_position=kwargs.get("cache_position"),
            )
        elif self.block_type == "attention":
            hidden_states = self.mixer(
                hidden_states, cache_position=kwargs.get("cache_position")
            )
            hidden_states = hidden_states[0]
        elif self.block_type == "mlp":
            hidden_states = self.mixer(hidden_states)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        hidden_states = residual + hidden_states
        return hidden_states

    block_cls.forward = patched_forward

    backbone = model.backbone if hasattr(model, "backbone") else model.model
    backbone_cls = type(backbone)
    backbone_cls._update_causal_mask = lambda self, *args, **kwargs: None


class ModelVariant(StrEnum):
    """Available Nemotron-H model variants for causal language modeling."""

    NEMOTRON_H_8B_BASE_8K = "H_8B_Base_8K"


class ModelLoader(ForgeModel):
    """Nemotron-H model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_H_8B_BASE_8K: LLMModelConfig(
            pretrained_model_name="nvidia/Nemotron-H-8B-Base-8K",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_H_8B_BASE_8K

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nemotron-H",
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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        _patch_model_for_cpu(model)

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
