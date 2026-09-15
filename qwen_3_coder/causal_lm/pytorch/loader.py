# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 Coder model loader implementation for causal language modeling.

Qwen3-Coder-30B-A3B-Instruct is a ``Qwen3MoeForCausalLM`` (``model_type:
qwen3_moe``): 48 layers, hidden 2048, GQA with 32 query and 4 KV heads, and a
sparse MoE block on every layer (128 experts, top-8, moe_intermediate 768).
Architecturally it matches Qwen3-30B-A3B and differs only in context length, but
it needs its own shard map: the ``qwen_3`` loader assumes a dense
``mlp.gate_proj/up_proj/down_proj``, which a ``Qwen3MoeSparseMoeBlock`` does not
have.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available Qwen 3 Coder MoE variants for causal language modeling."""

    QWEN_3_CODER_30B_A3B_INSTRUCT = "30B_A3B_Instruct"


class ModelLoader(ForgeModel):
    """Qwen 3 Coder model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_CODER_30B_A3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_CODER_30B_A3B_INSTRUCT

    sample_text = "Write a Python function that reverses a linked list."

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
            model="Qwen 3 Coder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Without this the forward output also carries a Cache, which the
        # runner's pytree comparator cannot diff leaf-wise against the CPU
        # golden. Set after load so it wins over anything passed in **kwargs.
        model.config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(
        self, dtype_override=None, prompt: Optional[str] = None, batch_size=1
    ):
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [{"role": "user", "content": prompt or self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)

        # Query and KV projections are both sharded on the model axis, so a mesh
        # wider than num_key_value_heads (4) would split a KV head.
        cfg = self.config or self.load_config()
        assert (
            cfg.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        assert (
            cfg.num_key_value_heads % mesh_shape[1] == 0
        ), "KV heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP map: column-parallel q/k/v, row-parallel o_proj.

        Every layer is a ``Qwen3MoeSparseMoeBlock``, so there is no dense MLP to
        shard. The routed experts' fused weights (``mlp.experts.gate_up_proj`` /
        ``down_proj``) are sharded on the expert dimension by the runner's
        ``get_tt_moe_shard_specs``; the router (``mlp.gate.weight``) stays
        replicated so every device can score all 128 experts before dispatch.
        """
        shard_specs = {}

        for layer in model.model.layers:
            sa = layer.self_attn
            shard_specs[sa.q_proj.weight] = ("model", "batch")
            shard_specs[sa.k_proj.weight] = ("model", "batch")
            shard_specs[sa.v_proj.weight] = ("model", "batch")
            shard_specs[sa.o_proj.weight] = ("batch", "model")

        shard_specs[model.model.embed_tokens.weight] = ("model", "batch")
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs
