# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Olmo3-7B-exp1b-lora128-e5 causal LM model loader implementation.

This is a LoRA adapter on allenai/Olmo-3-1025-7B from eac123/olmo3-7b-exp1b-lora128-e5.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
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
    """Available Olmo3-7B-exp1b-lora128-e5 model variants for causal language modeling."""

    Olmo3_7B_Exp1B_LoRA128_E5 = "olmo3_7b_exp1b_lora128_e5"


class ModelLoader(ForgeModel):
    """Olmo3-7B-exp1b-lora128-e5 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.Olmo3_7B_Exp1B_LoRA128_E5: LLMModelConfig(
            pretrained_model_name="eac123/olmo3-7b-exp1b-lora128-e5",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Olmo3_7B_Exp1B_LoRA128_E5

    BASE_MODEL_NAME = "allenai/Olmo-3-1025-7B"

    sample_text = "Who would win in a fight - a dinosaur or a cow named Moo Moo?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="olmo3_7b_exp1b_lora128_e5",
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
            self.BASE_MODEL_NAME, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        if getattr(model.config, "use_cache", True):
            model.config.layer_types = [
                "full_attention"
            ] * model.config.num_hidden_layers
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

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

    def get_mesh_config(self, num_devices: int):
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("batch", "model")

        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)

        return self.config
