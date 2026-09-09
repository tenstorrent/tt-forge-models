# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 model loader implementation for causal language modeling.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....tools.conv_overrides import slice_depthwise_conv1d_channels
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
    """Available Qwen 3.6 MoE model variants for causal language modeling."""

    QWEN_3_6_35B_A3B = "35B_A3B"


class ModelLoader(ForgeModel):
    """Qwen 3.6 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-35B-A3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_35B_A3B

    sample_text = "Give me a short introduction to large language model."

    # Gated DeltaNet conv is 8192 wide; 4096 and 1024 both verified on a
    # 4-device blackhole mesh. Lower it if the DRAM auto-slice assert returns.
    CONV_CHANNEL_CHUNK = 4096

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
            model="Qwen 3.6",
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
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}

        if self.num_layers is not None:
            # Qwen 3.6 keeps the decoder depth in the nested text_config; setting
            # it on the outer config is ignored (the model still builds all 64
            # layers). Set text_config and keep layer_types consistent so the
            # hybrid linear/full pattern still includes a full_attention layer.
            config = AutoConfig.from_pretrained(pretrained_model_name)
            text_cfg = getattr(config, "text_config", config)
            text_cfg.num_hidden_layers = self.num_layers
            if getattr(text_cfg, "layer_types", None) is not None:
                text_cfg.layer_types = text_cfg.layer_types[: self.num_layers]
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output is logits only. With the
        # checkpoint default (use_cache=True) it also carries a DynamicCache,
        # which the runner's pytree comparator can't diff leaf-wise against the
        # CPU golden. Set after load rather than via from_pretrained kwargs so
        # it wins regardless of what the caller passed in **kwargs.
        model.config.use_cache = False

        slice_depthwise_conv1d_channels(model, self.CONV_CHANNEL_CHUNK)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(
        self, dtype_override=None, prompt: Optional[str] = None, batch_size=1
    ):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": prompt or self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}

        for layer in model.model.layers:
            # Every layer is MoE: the routed experts' fused weights
            # (mlp.experts.gate_up_proj / down_proj) are sharded on the expert
            # dimension by get_tt_moe_shard_specs. The router (mlp.gate.weight)
            # and shared_expert_gate stay replicated so every device can score
            # all 256 experts before dispatch. The always-on shared expert is a
            # dense MLP: column-parallel gate/up, row-parallel down.
            shared = layer.mlp.shared_expert
            shard_specs[shared.gate_proj.weight] = ("model", "batch")
            shard_specs[shared.up_proj.weight] = ("model", "batch")
            shard_specs[shared.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("batch", "model")
                shard_specs[sa.k_proj.weight] = ("batch", "model")
                shard_specs[sa.v_proj.weight] = ("batch", "model")
                shard_specs[sa.o_proj.weight] = ("model", "batch")

            elif hasattr(layer, "linear_attn"):
                la = layer.linear_attn
                shard_specs[la.in_proj_qkv.weight] = ("model", "batch")
                if hasattr(la, "conv1d"):
                    shard_specs[la.conv1d.weight] = (None, None, None)
                shard_specs[la.in_proj_z.weight] = ("model", "batch")
                shard_specs[la.in_proj_a.weight] = ("model", "batch")
                shard_specs[la.in_proj_b.weight] = ("model", "batch")
                shard_specs[la.out_proj.weight] = ("batch", "model")
                if hasattr(la, "dt_bias"):
                    shard_specs[la.dt_bias] = ("model",)
                if hasattr(la, "A_log"):
                    shard_specs[la.A_log] = ("model",)

        shard_specs[model.model.embed_tokens.weight] = ("model", "batch")
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_activation_shard_spec(self, model):
        """Sharding constraints for intermediate ACTIVATIONS (not weights).

        The gated-delta block's fused ``in_proj_qkv`` is sharded contiguously on
        the "model" axis; the subsequent ``torch.split`` into [Q, K, V] cuts that
        sharded axis at points that don't align with the per-device boundaries,
        which miscompiles under Shardy and scrambles q/k/v before the recurrence
        (full-model PCC collapses). Replicating the conv output before the split
        makes the split run on correct data.
        """
        constraints = {}
        for layer in model.model.layers:
            if layer.layer_type == "linear_attention":
                constraints[layer.linear_attn.conv1d] = None
        return constraints
