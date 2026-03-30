# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.2 model loader implementation.

Uses a locally modified Transformer (model.py) instead of the original
HuggingFace model. The modifications are:
  1. Uses scipy.linalg.hadamard instead of fast_hadamard_transform (no CUDA required).
  2. Stubs out FP8 quantization (act_quant, fp8_gemm, fp8_index) that rely on
     custom tilelang kernels unsupported on TT hardware.
  3. Avoids torch.view_as_complex / view_as_real operations.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PretrainedConfig

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

    DEEPSEEK_V3_2_EXP = "deepseek_v3_2_exp"


from .modified_model import ModelArgs, Transformer


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

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> _CausalLMOutput:
        # start_pos is fixed at 0 for TT compilation compatibility.
        # Any Python-level computation before self.transformer() that touches
        # tensor values (cache_position[0].item()) or mutates self attributes
        # with SymInts (self._current_pos += seqlen) causes TorchDynamo to split
        # the forward into two sub-graphs.  The second sub-graph (the transformer
        # call) is then compiled in isolation by the TT backend and crashes with
        # SIGFPE during MLIR lowering.  Using a constant 0 keeps the entire
        # forward as a single compiled graph, matching the behaviour of the
        # working unit test (test_deepseek_modified_transformer_single_layer).
        # TODO: support proper KV-cache position tracking once TT compilation
        # handles dynamic start_pos.
        logits = self.transformer(tokens=input_ids, start_pos=0)
        return _CausalLMOutput(logits=logits.unsqueeze(1))


class ModelLoader(ForgeModel):
    """DeepSeek V3.2 model loader using the locally modified Transformer."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_V3_2_EXP: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-V3.2-Exp",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_V3_2_EXP

    def __init__(
        self,
        variant=None,
        num_layers: Optional[int] = None,
        max_batch_size: int = 32,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional variant string. Unused; kept for API compatibility.
            num_layers: Number of transformer layers to instantiate.
                        If None, uses the ModelArgs default (27).
            max_batch_size: Maximum batch size for KV-cache allocation.
                            Must be >= the batch size used at inference time.
                            Defaults to 32 to match the benchmark default.
        """
        super().__init__(variant)
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.tokenizer = None
        self.model = None
        # self.config = None

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
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def _load_config(self, args=None):
        """Load config from the local config.json, then overlay args-derived values.

        The JSON file provides the full-model defaults from the original DeepSeek
        V3.2 checkpoint.  When *args* is provided, any fields that depend on the
        actually-instantiated model (num_hidden_layers, num_attention_heads,
        hidden_size, head_dim) are overridden so they stay consistent with the
        running model — e.g. when num_layers is reduced for bringup tests.

        The JSON key ``v_head_dim`` is remapped to ``head_dim`` because that is
        the name expected by HuggingFace utilities such as ``StaticCache``.

        Args:
            args: Optional ModelArgs instance whose values take precedence over
                  the JSON defaults for architecture-critical fields.

        Returns:
            PretrainedConfig: Populated config stored on ``self.config``.
        """
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            raw = json.load(f)

        # Remap v_head_dim → head_dim (HuggingFace convention for StaticCache).
        if "v_head_dim" in raw:
            raw.setdefault("head_dim", raw.pop("v_head_dim"))

        config = PretrainedConfig(**raw)

        # Override with values from the instantiated ModelArgs so that
        # architecture-critical fields reflect the actual running model.
        if args is not None:
            config.num_hidden_layers = args.n_layers
            config.num_attention_heads = args.n_heads
            config.num_key_value_heads = args.n_heads
            config.hidden_size = args.dim
            config.head_dim = args.v_head_dim

        self.config = config
        return config

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

        # Seed ModelArgs from config.json so that all V3.2-specific architecture
        # values are correct.  ModelArgs dataclass defaults are for a smaller
        # (DeepSeek-V2-scale) model; without seeding, e.g. dim=2048 instead of
        # 7168, n_routed_experts=64 instead of 256, etc.  Caller kwargs and the
        # num_layers override take precedence over these JSON-derived defaults.
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as _f:
            _cfg = json.load(_f)
        # Maps config.json keys → ModelArgs field names.
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
            args_key: _cfg[json_key]
            for json_key, args_key in _JSON_TO_ARGS.items()
            if json_key in _cfg
        }
        # Caller kwargs override JSON defaults.
        model_args_kwargs.update(kwargs)
        # num_layers overrides n_layers last so it always wins.
        # n_dense_layers comes from config.json (first_k_dense_replace=3), giving
        # layers 0-2 as dense MLP and layers 3+ as MoE for a 5-layer test.
        if self.num_layers is not None:
            model_args_kwargs["n_layers"] = self.num_layers
        model_args_kwargs.setdefault("max_batch_size", self.max_batch_size)
        args = ModelArgs(**model_args_kwargs)

        transformer = Transformer(args)

        if dtype_override is not None:
            transformer = transformer.to(dtype_override)

        transformer = transformer.eval()
        self._args = args

        config = self._load_config(args=args)

        return DeepSeekV32ForCausalLM(transformer, config)

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

        Sharding convention (Galaxy mesh: batch=4, model=8):
          ColumnParallelLinear weight [out, in]  → ("model", "batch")
          RowParallelLinear weight    [out, in]  → ("batch", "model")
          Input-only projections      [out, in]  → (None, "batch")
          KV / position caches       [B, S, D]  → ("batch", None, None)
          1-D norm weights                       → (None,)  [replicated]

        MoE expert weights are left un-specified (replicated) because expert
        parallelism requires a separate partitioning strategy.

        Args:
            model: DeepSeekV32ForCausalLM wrapper instance (on device).

        Returns:
            dict mapping parameter/buffer tensors to shard-spec tuples.
        """
        from .modified_model import MLP, MoE

        shard_specs = {}
        t = model.transformer

        # ── Global tensors ────────────────────────────────────────────────────
        shard_specs[t.embed.weight] = (None, "batch")  # [vocab, dim]
        shard_specs[t.norm.weight] = (None,)  # [dim]
        shard_specs[t.head.weight] = ("model", "batch")  # [vocab, dim]

        for layer in t.layers:
            attn = layer.attn

            # ── MLA attention ─────────────────────────────────────────────────
            # Input projections (not column-parallel; shard on input dim)
            shard_specs[attn.wq_a.weight] = (None, "batch")  # [q_lora_rank, dim]
            shard_specs[attn.wkv_a.weight] = (
                None,
                "batch",
            )  # [kv_lora_rank+rope_hd, dim]
            # Norm weights (1-D) — replicated
            shard_specs[attn.q_norm.weight] = (None,)
            shard_specs[attn.kv_norm.weight] = (None,)
            # Column-parallel Q and KV up-projections
            shard_specs[attn.wq_b.weight] = (
                "model",
                None,
            )  # [n_heads*qk_hd, q_lora_rank]
            shard_specs[attn.wkv_b.weight] = (
                "model",
                None,
            )  # [n_heads*(nope+v_hd), kv_lora_rank]
            # Row-parallel output projection
            shard_specs[attn.wo.weight] = (
                "batch",
                "model",
            )  # [dim, n_heads*v_head_dim]
            # KV and positional caches — shard on batch dimension
            shard_specs[attn.kv_cache] = ("batch", None, None)
            shard_specs[attn.pe_cache] = ("batch", None, None)

            # ── Indexer (token-selection attention) ───────────────────────────
            if attn.indexer is not None:
                idx = attn.indexer
                shard_specs[idx.wq_b.weight] = (
                    "model",
                    None,
                )  # [n_idx_heads*idx_hd, q_lora_rank]
                shard_specs[idx.wk.weight] = (None, "batch")  # [idx_hd, dim]
                shard_specs[idx.k_norm.weight] = (None,)  # [idx_hd]
                shard_specs[idx.k_norm.bias] = (None,)  # [idx_hd]
                shard_specs[idx.weights_proj.weight] = (
                    "model",
                    "batch",
                )  # [n_idx_heads, dim]
                shard_specs[idx.k_cache] = ("batch", None, None)

            # ── FFN ───────────────────────────────────────────────────────────
            ffn = layer.ffn
            if isinstance(ffn, MLP):
                shard_specs[ffn.w1.weight] = (
                    "model",
                    "batch",
                )  # gate  [inter_dim, dim]
                shard_specs[ffn.w3.weight] = (
                    "model",
                    "batch",
                )  # up    [inter_dim, dim]
                shard_specs[ffn.w2.weight] = (
                    "batch",
                    "model",
                )  # down  [dim, inter_dim]
            elif isinstance(ffn, MoE):
                # Gate routing weights — replicated; routing decisions are
                # small and must be consistent across all devices.
                shard_specs[ffn.gate.weight] = (None, "batch")  # [n_experts, dim]
                if ffn.gate.bias is not None:
                    shard_specs[ffn.gate.bias] = (None,)

                # Routed experts — shard each expert's weights the same way
                # as a dense MLP (Megatron column/row parallel pattern).
                for expert in ffn.experts:
                    if expert is None:
                        continue
                    shard_specs[expert.w1.weight] = (
                        "model",
                        "batch",
                    )  # [moe_inter_dim, dim]
                    shard_specs[expert.w3.weight] = (
                        "model",
                        "batch",
                    )  # [moe_inter_dim, dim]
                    shard_specs[expert.w2.weight] = (
                        "batch",
                        "model",
                    )  # [dim, moe_inter_dim]

                # Shared expert — treated as a regular MLP.
                shard_specs[ffn.shared_experts.w1.weight] = ("model", "batch")
                shard_specs[ffn.shared_experts.w3.weight] = ("model", "batch")
                shard_specs[ffn.shared_experts.w2.weight] = ("batch", "model")

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
