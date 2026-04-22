# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral-Small-4-119B-2603 NVFP4 model loader implementation for causal language modeling.

The NVFP4 checkpoint uses Mistral's native format (params.json, no config.json)
with NVIDIA FP4 quantization.  For compilation testing we load the architecture
from the BF16 counterpart (mistralai/Mistral-Small-4-119B-2603) which exposes a
standard HuggingFace config, extract its Mistral4TextConfig, reduce the layer
count and hidden dimensions to a tractable size, and initialise the model with
random weights via Mistral4ForCausalLM(config).  The tokenizer is loaded from
the NVFP4 repo directly, which does ship a tokenizer.json.
"""
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer, Mistral4ForCausalLM

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

# BF16 companion repo that carries a standard HuggingFace config.json.
_CONFIG_REPO = "mistralai/Mistral-Small-4-119B-2603"


class ModelVariant(StrEnum):
    """Available Mistral-Small-4-119B-2603 NVFP4 model variants."""

    MISTRAL_SMALL_4_119B_2603_NVFP4 = "119B_2603_NVFP4"


class ModelLoader(ForgeModel):
    """Mistral-Small-4-119B-2603 NVFP4 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_4_119B_2603_NVFP4: LLMModelConfig(
            pretrained_model_name="mistralai/Mistral-Small-4-119B-2603-NVFP4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_4_119B_2603_NVFP4

    sample_text = "Give me a short introduction to large language models."

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
            model="Mistral-Small-4-119B-2603-NVFP4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_text_config(self):
        """Return a Mistral4Config with reduced dimensions for compile testing."""
        outer = AutoConfig.from_pretrained(_CONFIG_REPO)
        cfg = outer.text_config

        if self.num_layers is not None:
            cfg.num_hidden_layers = self.num_layers
        else:
            cfg.num_hidden_layers = 6

        cfg.hidden_size = 1024
        cfg.intermediate_size = 4096
        cfg.num_attention_heads = 8
        cfg.num_key_value_heads = 8
        cfg.kv_lora_rank = 128
        cfg.q_lora_rank = 256
        cfg.qk_head_dim = 128
        cfg.qk_nope_head_dim = 64
        cfg.qk_rope_head_dim = 64
        cfg.v_head_dim = 128
        cfg.n_routed_experts = 8
        cfg.num_experts_per_tok = 2
        cfg.n_shared_experts = 1
        cfg.moe_intermediate_size = 512

        return cfg

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        cfg = self._build_text_config()
        cfg._attn_implementation = "eager"

        model = Mistral4ForCausalLM(cfg).eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
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
        self.config = self._build_text_config()
        return self.config

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            # MLA attention projections
            shard_specs[layer.self_attn.q_b_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.kv_b_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            # Shared (dense) expert
            shared = layer.mlp.shared_experts
            shard_specs[shared.up_proj.weight] = ("model", "batch")
            shard_specs[shared.gate_proj.weight] = ("model", "batch")
            shard_specs[shared.down_proj.weight] = ("batch", "model")

            # Routed (sparse) expert weight tensors
            shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch")
            shard_specs[layer.mlp.experts.down_proj] = ("batch", "model")

        return shard_specs
