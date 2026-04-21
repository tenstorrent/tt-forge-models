# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Small 4 119B GGUF model loader implementation for causal language modeling.
"""

import os
from typing import Optional

import torch
from transformers import AutoTokenizer, Mistral4Config, Mistral4ForCausalLM

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Mistral Small 4 119B GGUF model variants for causal language modeling."""

    MISTRAL_SMALL_4_119B_GGUF = "Small_4_119B_GGUF"


class ModelLoader(ForgeModel):
    """Mistral Small 4 119B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_4_119B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_4_119B_GGUF

    GGUF_FILE = "mistralai_Mistral-Small-4-119B-2603-Q4_K_M/mistralai_Mistral-Small-4-119B-2603-Q4_K_M-00001-of-00002.gguf"

    BASE_MODEL = "mistralai/Mistral-Small-4-119B-2603"

    sample_text = "What is your favorite city?"

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
            model="Mistral Small 4 119B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_config(self):
        config = Mistral4Config(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=1,
            vocab_size=131072,
            max_position_embeddings=1048576,
            rms_norm_eps=1e-6,
            n_routed_experts=128,
            num_experts_per_tok=4,
            n_shared_experts=1,
            moe_intermediate_size=2048,
            q_lora_rank=1024,
            kv_lora_rank=256,
            qk_head_dim=128,
            v_head_dim=128,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
            rope_interleave=True,
            rope_parameters={
                "type": "yarn",
                "rope_theta": 10000.0,
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "max_position_embeddings": 1048576,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
                "partial_rotary_factor": 0.5,
                "rope_type": "yarn",
            },
        )
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        return config

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self._build_config()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if os.environ.get("TT_RANDOM_WEIGHTS") == "1":
            model = Mistral4ForCausalLM(config).to(
                dtype=model_kwargs.get("torch_dtype", torch.float32)
            )
        else:
            model_kwargs["gguf_file"] = self.GGUF_FILE
            model_kwargs |= kwargs
            model_kwargs["config"] = config
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self._variant_config.pretrained_model_name, **model_kwargs
            )

        model = model.eval()
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
            shard_specs[layer.self_attn.q_a_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_b_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = ("model", "batch")
            shard_specs[layer.self_attn.kv_b_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.shared_experts.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.shared_experts.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.shared_experts.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = self._build_config()
        return self.config
