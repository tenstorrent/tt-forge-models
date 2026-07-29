# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 model loader implementation for causal language modeling.

Qwen 3.5 uses a hybrid architecture interleaving Gated DeltaNet (linear
attention with causal conv1d + chunked delta rule) and standard full
attention layers. Dense variants follow the layout
(3x linear_attention + 1x full_attention) repeated.
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


class ModelVariant(StrEnum):
    """Available Qwen 3.5 dense model variants for causal language modeling."""

    QWEN_3_5_0_8B = "0_8B"
    QWEN_3_5_2B = "2B"
    QWEN_3_5_4B = "4B"
    QWEN_3_5_9B = "9B"
    QWEN_3_5_27B = "27B"
    QWEN_3_5_122B_A10B = "122B_A10B"


# Variants promoted to the RED model group.
_RED_VARIANTS = {
    ModelVariant.QWEN_3_5_27B,
    ModelVariant.QWEN_3_5_122B_A10B,
}


class ModelLoader(ForgeModel):
    """Qwen 3.5 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_0_8B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-0.8B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_2B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-2B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_4B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-4B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_9B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-9B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-27B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_122B_A10B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-122B-A10B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_0_8B

    sample_text = "Give me a short introduction to large language model."

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

        group = ModelGroup.RED if variant in _RED_VARIANTS else ModelGroup.GENERALITY

        return ModelInfo(
            model="Qwen 3.5",
            variant=variant,
            group=group,
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
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            # Qwen 3.5 keeps the decoder depth in the nested text_config; setting
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

        # Force use_cache=False on the live model config so the forward
        # output does not include a Qwen3_5DynamicCache, which the runner's
        # pytree comparator can't diff leaf-wise against the CPU golden.
        # Same pattern as qwen_2_5_vl loader — passing use_cache via
        # from_pretrained kwargs / config is overwritten when the model
        # rebuilds its config from the checkpoint.
        model.config.use_cache = False

        if self._variant == ModelVariant.QWEN_3_5_122B_A10B:
            # Duplicate GQA KV heads up to 8 so full-attention can be
            # head-parallel-sharded on the galaxy "model" axis (size 8). Qwen3.5
            # has only 2 KV heads, which don't tile 8; the resulting hidden-dim
            # sharding is what forced the batch<->model residual reshards
            # (sdy.collective_permute, tt-mlir #3370). Padding lets attention use
            # the gpt_oss-style head-parallel layout with a uniform residual axis.
            self._pad_kv_heads(model, target_kv_heads=8)

        self.config = model.config
        self.model = model
        return model

    def _pad_kv_heads(self, model, target_kv_heads=8):
        """Duplicate GQA key/value heads up to ``target_kv_heads``.

        Numerically identical: GQA already broadcasts each KV head to a fixed
        group of query heads, so duplicating a KV head (contiguously, matching
        ``repeat_kv``) and shrinking the group size leaves attention unchanged.
        Only the weights, ``num_key_value_groups`` and the config are touched --
        the attention ``forward`` infers the head count from the tensor width,
        so no method override is needed.
        """
        text_cfg = getattr(model.config, "text_config", model.config)
        orig = text_cfg.num_key_value_heads
        if orig >= target_kv_heads or target_kv_heads % orig != 0:
            return
        rep = target_kv_heads // orig
        head_dim = getattr(
            text_cfg,
            "head_dim",
            text_cfg.hidden_size // text_cfg.num_attention_heads,
        )
        for layer in model.model.layers:
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
            for name in ("k_proj", "v_proj"):
                proj = getattr(sa, name)
                in_f = proj.weight.shape[1]
                w = (
                    proj.weight.data.view(orig, head_dim, in_f)
                    .repeat_interleave(rep, dim=0)
                    .reshape(target_kv_heads * head_dim, in_f)
                )
                proj.weight = torch.nn.Parameter(w, requires_grad=False)
                proj.out_features = target_kv_heads * head_dim
                if proj.bias is not None:
                    b = (
                        proj.bias.data.view(orig, head_dim)
                        .repeat_interleave(rep, dim=0)
                        .reshape(-1)
                    )
                    proj.bias = torch.nn.Parameter(b, requires_grad=False)
            sa.num_key_value_groups = text_cfg.num_attention_heads // target_kv_heads
            if hasattr(sa, "num_key_value_heads"):
                sa.num_key_value_heads = target_kv_heads
        text_cfg.num_key_value_heads = target_kv_heads

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

    def load_inputs_decode(self, dtype_override=None, batch_size=1):
        from ....tools.utils import get_static_cache_decode_inputs

        if self.tokenizer is None:
            self._load_tokenizer()
        if self.config is None:
            self.load_config()

        max_cache_len = getattr(self._variant_config, "max_length", None) or 128
        self.seq_len = 1

        return get_static_cache_decode_inputs(
            tokenizer=self.tokenizer,
            config=self.config,
            model=self.model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype_override,
        )

    def get_mesh_config(self, num_devices: int):
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}

        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                # MoE layer (122B-A10B): routed experts are sharded on the
                # expert dimension by get_tt_moe_shard_specs (inject_custom_moe).
                # The always-on shared expert is a dense MLP: keep its hidden
                # dim on "batch" (gate/up input, down output) to stay consistent
                # with the residual stream.
                shared = mlp.shared_expert
                shard_specs[shared.gate_proj.weight] = ("model", "batch")
                shard_specs[shared.up_proj.weight] = ("model", "batch")
                shard_specs[shared.down_proj.weight] = ("batch", "model")
            else:
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                sa = layer.self_attn
                if self._variant == ModelVariant.QWEN_3_5_122B_A10B:
                    # KV heads padded to 8 (see _pad_kv_heads) so attention is
                    # head-parallel like gpt_oss: heads sharded on "model",
                    # hidden contracted on "batch". This keeps the residual on
                    # "batch" through attention too (matching MLP/gated-delta),
                    # avoiding the batch<->model reshard that Shardy lowers to
                    # sdy.collective_permute on the 2D galaxy mesh.
                    shard_specs[sa.q_proj.weight] = ("model", "batch")
                    shard_specs[sa.k_proj.weight] = ("model", "batch")
                    shard_specs[sa.v_proj.weight] = ("model", "batch")
                    shard_specs[sa.o_proj.weight] = ("batch", "model")
                else:
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
