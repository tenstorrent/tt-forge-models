# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UnifiedReward-Edit-qwen-72b i1 GGUF model loader implementation for image to text.

Note: The qwen2vl GGUF architecture is not yet supported by the transformers
GGUF loader, and dequantizing 963 GGUF tensors for a 72B model takes ~40
minutes on CPU. We load the Qwen2 text decoder config from the base model
and instantiate with random weights for compile-only testing.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
    """Available UnifiedReward-Edit-qwen-72b i1 GGUF model variants for image to text."""

    UNIFIED_REWARD_EDIT_QWEN_72B_I1_GGUF = "72b_i1_gguf"


class ModelLoader(ForgeModel):
    """UnifiedReward-Edit-qwen-72b i1 GGUF model loader implementation for image to text tasks.

    Uses the Qwen2 text decoder config from the base model with random weights
    for compile-only testing. The qwen2vl GGUF architecture is not supported
    by the transformers GGUF loader, and full GGUF dequantization of a 72B
    model takes ~40 minutes on CPU, making it impractical for compile-only runs.
    """

    _VARIANTS = {
        ModelVariant.UNIFIED_REWARD_EDIT_QWEN_72B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/UnifiedReward-Edit-qwen-72b-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNIFIED_REWARD_EDIT_QWEN_72B_I1_GGUF

    BASE_MODEL = "CodeGoat24/UnifiedReward-Edit-qwen-72b"

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="UnifiedReward-Edit-qwen-72b i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        full_config = AutoConfig.from_pretrained(self.BASE_MODEL)
        config = full_config.text_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)[
                    :batch_size
                ]

        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
