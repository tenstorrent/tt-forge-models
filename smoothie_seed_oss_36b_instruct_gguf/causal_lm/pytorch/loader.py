# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Smoothie-Seed-OSS-36B-Instruct GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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

BASE_MODEL = "ByteDance-Seed/Seed-OSS-36B-Instruct"


class ModelVariant(StrEnum):
    """Available Smoothie-Seed-OSS-36B-Instruct GGUF model variants for causal language modeling."""

    SMOOTHIE_SEED_OSS_36B_INSTRUCT_I1_GGUF = "Smoothie-Seed-OSS-36B-Instruct-i1-GGUF"


class ModelLoader(ForgeModel):
    """Smoothie-Seed-OSS-36B-Instruct GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SMOOTHIE_SEED_OSS_36B_INSTRUCT_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Smoothie-Seed-OSS-36B-Instruct-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOOTHIE_SEED_OSS_36B_INSTRUCT_I1_GGUF

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Smoothie-Seed-OSS-36B-Instruct GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # The transformers GGUF loader does not yet support the seed_oss architecture,
        # so we load config from the base model and instantiate with random weights.
        config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        target_dtype = dtype_override if dtype_override is not None else torch.float32
        old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(target_dtype)
        try:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        finally:
            torch.set_default_dtype(old_default_dtype)
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
        return self.config
