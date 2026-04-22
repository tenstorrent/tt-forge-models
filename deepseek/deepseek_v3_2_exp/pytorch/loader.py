# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V3.2 model loader using the locally modified Transformer from src/modified_model.py."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PretrainedConfig
import torch_xla.runtime as xr
from tt_torch.sparse_mlp import enable_sparse_mlp

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
    LLMModelConfig,
)


class ModelVariant(StrEnum):
    """Available DeepSeek V3.2 model variants."""

    DEEPSEEK_V3_2_EXP_MODIFIED = "deepseek_v3_2_exp_modified"


from .src.modified_model import LayerNorm, MoE, ModelArgs, Transformer


@dataclass
class _CausalLMOutput:
    """Minimal output container satisfying ``output.logits`` expected by the benchmark."""

    logits: torch.Tensor


class DeepSeekV32ForCausalLM(nn.Module):
    """HuggingFace-compatible wrapper around the custom Transformer.

    The benchmark calls ``model(input_ids=..., past_key_values=...,
    cache_position=..., use_cache=...)`` and reads ``output.logits``.
    The custom Transformer uses ``forward(tokens, start_pos)`` and returns a
    raw ``[batch, vocab_size]`` tensor.  This wrapper bridges that gap without
    touching ``modified_model.py``.

    ``cache_position`` is ``[seq_len]`` on the prefill step and ``[1]`` on
    each decode step; its first value is the absolute position of the first
    token being processed, which maps directly to ``start_pos`` in the custom
    model's rotary-embedding indexing.

    The logits are unsqueezed from ``[batch, vocab]`` to ``[batch, 1, vocab]``
    so that ``logits[:, -1].argmax(dim=-1)`` in the decode loop works correctly.
    """

    def __init__(self, transformer: Transformer, config: PretrainedConfig):
        super().__init__()
        self.transformer = transformer
        self.config = config
        # Pre-allocate so forward slices a buffer rather than calling torch.full(..., device=...),
        # which causes a TorchDynamo graph break that splits the compiled SPMD graph.
        max_seq_len = transformer.freqs_cis.size(0)
        causal_mask = torch.full(
            (max_seq_len, max_seq_len), float("-inf"), dtype=torch.bfloat16
        ).triu_(1)
        self.register_buffer("_causal_mask", causal_mask, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> _CausalLMOutput:
        # start_pos=0: cache_position[0].item() would cause a TorchDynamo graph break splitting
        # the SPMD graph; freqs_cis[cache_position] is a no-.item() gather that avoids this.
        seqlen = input_ids.size(1)
        start_pos = 0
        if cache_position is not None:
            freqs_cis = self.transformer.freqs_cis[cache_position]
        else:
            freqs_cis = self.transformer.freqs_cis[:seqlen]
        mask = self._causal_mask[:seqlen, :seqlen] if seqlen > 1 else None
        logits = self.transformer(
            tokens=input_ids,
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            mask=mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        return _CausalLMOutput(logits=logits.unsqueeze(1))


class ModelLoader(ForgeModel):
    """DeepSeek V3.2 model loader using the locally modified Transformer."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_V3_2_EXP_MODIFIED: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-V3.2-Exp",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_V3_2_EXP_MODIFIED

    def __init__(
        self,
        variant=None,
        num_layers: Optional[int] = None,
        max_batch_size: int = 128,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional variant string. Unused; kept for API compatibility.
            num_layers: Number of transformer layers to instantiate.
                        If None, uses the ModelArgs default (27).
            max_batch_size: Maximum batch size for KV-cache allocation.
                            Must be >= the batch size used at inference time.
                            Defaults to 128 to match the galaxy benchmark batch size.
        """
        super().__init__(variant)
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size  # TODO
        self.tokenizer = None
        self.model = None
        self.config = None
        self._args = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Return model metadata for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3.2",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def _load_config(self, **kwargs):
        """Load src/config.json, build ModelArgs, and populate self.config and self._args.

        Sets self.config (PretrainedConfig for HuggingFace compatibility) and
        self._args (ModelArgs for the Transformer), with architecture-critical
        fields on self.config overridden to match the instantiated model.

        Args:
            **kwargs: Forwarded to ModelArgs, overriding JSON defaults.

        Returns:
            PretrainedConfig: Populated config stored on ``self.config``.
        """
        config_path = Path(__file__).parent / "src" / "config.json"
        with open(config_path) as f:
            raw = json.load(f)

        self.config = PretrainedConfig(**raw)

        _JSON_TO_ARGS = {
            "hidden_size": "dim",
            "intermediate_size": "inter_dim",
            "moe_intermediate_size": "moe_inter_dim",
            "num_hidden_layers": "n_layers",
            "first_k_dense_replace": "n_dense_layers",
            "num_attention_heads": "n_heads",
            "n_routed_experts": "n_routed_experts",
            "n_shared_experts": "n_shared_experts",
            "num_experts_per_tok": "n_activated_experts",
            "n_group": "n_expert_groups",
            "topk_group": "n_limited_groups",
            "scoring_func": "score_func",
            "routed_scaling_factor": "route_scale",
            "q_lora_rank": "q_lora_rank",
            "kv_lora_rank": "kv_lora_rank",
            "qk_nope_head_dim": "qk_nope_head_dim",
            "qk_rope_head_dim": "qk_rope_head_dim",
            "v_head_dim": "v_head_dim",
            "vocab_size": "vocab_size",
            "rope_theta": "rope_theta",
            "index_n_heads": "index_n_heads",
            "index_head_dim": "index_head_dim",
            "index_topk": "index_topk",
        }
        model_args_kwargs = {
            args_key: raw[json_key]
            for json_key, args_key in _JSON_TO_ARGS.items()
            if json_key in raw
        }
        model_args_kwargs.update(kwargs)
        if self.num_layers is not None:
            model_args_kwargs["n_layers"] = self.num_layers
            n_dense = model_args_kwargs.get("n_dense_layers", 0)
            if n_dense >= self.num_layers:
                model_args_kwargs["n_dense_layers"] = max(0, self.num_layers - 1)
        model_args_kwargs.setdefault("max_batch_size", self.max_batch_size)
        model_args_kwargs.setdefault("use_mla_cache", True)

        self._args = ModelArgs(**model_args_kwargs)

        # Keep HuggingFace config fields consistent with the instantiated model.
        self.config.num_hidden_layers = self._args.n_layers
        self.config.num_attention_heads = self._args.n_heads
        self.config.num_key_value_heads = self._args.n_heads
        self.config.hidden_size = self._args.dim
        self.config.head_dim = self._args.v_head_dim
        self.config.index_topk = self._args.index_topk

        return self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the modified DeepSeek V3.2 Transformer.

        The model is constructed from ModelArgs defaults, overriding n_layers
        with the value passed at construction time.

        Args:
            dtype_override: Optional torch.dtype to cast the model to after
                            construction (e.g. torch.bfloat16).

        Returns:
            torch.nn.Module: The modified DeepSeek V3.2 Transformer in eval mode.
        """

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if self.config is None or self._args is None:
            self._load_config(**kwargs)

        transformer = Transformer(self._args)

        if dtype_override is not None:
            transformer = transformer.to(dtype_override)

        # Restore float32 params that .to(bfloat16) silently converts.
        # head expects fp32 logits; LayerNorm calls x.float() internally and errors on mixed dtype.
        transformer.head = transformer.head.to(torch.float32)
        for module in transformer.modules():
            if isinstance(module, LayerNorm):
                module.weight.data = module.weight.data.to(torch.float32)
                module.bias.data = module.bias.data.to(torch.float32)

        has_moe = any(isinstance(layer.ffn, MoE) for layer in transformer.layers)
        if has_moe:
            num_devices = xr.global_runtime_device_count()
            mesh_shape, _ = self.get_mesh_config(num_devices)
            enable_sparse_mlp(
                transformer, mesh=mesh_shape, cluster_axis=0, config=self._args
            )

        transformer = transformer.eval()

        return DeepSeekV32ForCausalLM(transformer, self.config)

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallelism.

        Args:
            num_devices: Number of devices available at runtime.

        Returns:
            Tuple of (mesh_shape, axis_names).
        """
        if num_devices == 32:
            return (4, 8), ("batch", "model")
        raise ValueError(
            f"DeepSeek V3.2 is only supported on Galaxy (32 devices), got {num_devices}"
        )

    def load_shard_spec(self, model):
        """Build SPMD shard specifications for all model tensors.

        Sharding matches test_deepseek_v3_2_full_sparse_moe on a Galaxy (4, 8)
        mesh named ("model", "batch"):
          batch = axis 0, size 4  →  "model" in the unit test
          model = axis 1, size 8  →  "batch" in the unit test

        MLA attention weights shard on the batch axis (size 4).
        KV / positional caches shard on the model axis (size 8).
        Dense FFN weights shard on both axes.
        A2aSparse MoE experts shard on (batch, model) jointly.
        Layer norms shard on the batch axis.

        Args:
            model: DeepSeekV32ForCausalLM wrapper instance (on device).

        Returns:
            dict mapping parameter/buffer tensors to shard-spec tuples.
        """
        from .src.modified_model import MLP

        shard_specs = {}
        t = model.transformer

        shard_specs[t.embed.weight] = (None, "model")
        shard_specs[t.norm.weight] = ("model",)
        shard_specs[t.head.weight] = (None, "model")

        for layer in t.layers:
            attn = layer.attn

            shard_specs[attn.wq_a.weight] = (None, "model")
            shard_specs[attn.wkv_a.weight] = (None, "model")
            shard_specs[attn.wq_b.weight] = ("model", None)
            shard_specs[attn.wkv_b.weight] = ("model", None)
            shard_specs[attn.wo.weight] = (None, "model")

            # These local caches are only initialized when MLACache is not used
            if attn.kv_cache is not None:
                shard_specs[attn.kv_cache] = ("batch", None, None)
            if attn.pe_cache is not None:
                shard_specs[attn.pe_cache] = ("batch", None, None)

            if attn.indexer is not None:
                idx = attn.indexer
                shard_specs[idx.wq_b.weight] = ("model", None)
                shard_specs[idx.wk.weight] = (None, "model")
                shard_specs[idx.weights_proj.weight] = (None, "model")
                shard_specs[idx.k_cache] = ("batch", None, None, None)

            ffn = layer.ffn
            if hasattr(ffn, "mlp"):
                # A2aSparseMLPWithSharedExperts (MoE layer)
                mlp = ffn.mlp
                shard_specs[mlp.router.gate.weight] = (None, "model")
                shard_specs[mlp.experts.gate_proj] = (("model", "batch"), None, None)
                shard_specs[mlp.experts.up_proj] = (("model", "batch"), None, None)
                shard_specs[mlp.experts.down_proj] = (("model", "batch"), None, None)

                shared = getattr(ffn, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.w1.weight] = (None, "model")
                    shard_specs[shared.w3.weight] = (None, "model")
                    shard_specs[shared.w2.weight] = ("model", None)
            else:
                # Dense MLP
                shard_specs[ffn.w1.weight] = ("batch", "model")
                shard_specs[ffn.w3.weight] = ("batch", "model")
                shard_specs[ffn.w2.weight] = ("model", "batch")

            shard_specs[layer.attn_norm.weight] = ("model",)
            shard_specs[layer.ffn_norm.weight] = ("model",)

        return shard_specs

    def load_inputs(self, batch_size: int = 1, seq_len: int = 32):
        """Return sample token inputs for the model.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Length of each input sequence.

        Returns:
            torch.Tensor: Integer token tensor of shape (batch_size, seq_len).
        """
        if not hasattr(self, "_args"):
            self.load_model()

        tokens = torch.randint(0, self._args.vocab_size, (batch_size, seq_len))
        return tokens
