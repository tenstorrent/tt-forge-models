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

    DEEPSEEK_V3_2_EXP = "deepseek_v3_2_exp"


from .modified_model import LayerNorm, ModelArgs, Transformer


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
        # Pre-allocate a causal mask for the full max_seq_len so that forward()
        # can slice it rather than calling torch.full() dynamically.
        #
        # Computing freqs_cis and mask *inside* forward via torch.full(...,
        # device=tokens.device) causes TorchDynamo to emit a separate,
        # unsharded graph segment that runs before the SPMD device mesh is
        # opened.  That segment is compiled with a trivial [1,1] shardy mesh
        # and produces the wrong [1,N] tensor layout, causing a hang or SIGFPE
        # on device.  Pre-allocating here and slicing in forward matches the
        # approach used in test_deepseek_v3_2_full_sparse_moe, which
        # pre-computes both tensors outside torch.compile to keep the mesh
        # intact for the entire compiled graph.
        max_seq_len = transformer.freqs_cis.size(0)
        causal_mask = torch.full(
            (max_seq_len, max_seq_len), float("-inf"), dtype=torch.bfloat16
        ).triu_(1)
        self.register_buffer("_causal_mask", causal_mask, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> _CausalLMOutput:
        # When a DeepSeekMLACache is passed as past_key_values, use its
        # current_pos as start_pos and route cache writes/reads through it.
        # Otherwise fall back to start_pos=0 with internal model buffers.
        #
        # NOTE: start_pos must remain a Python int (not derived from a tensor
        # via .item()) to avoid TorchDynamo graph splits.  When past_key_values
        # is None, start_pos=0 is a compile-time constant.  When a
        # DeepSeekMLACache is provided, start_pos = cache.current_pos which is
        # also a Python int updated outside the compiled graph — this causes one
        # recompilation per unique start_pos value (each decode step).  Proper
        # compile-friendly position tracking is a future TODO.
        from tests.torch.models.utils.mla_cache import DeepSeekMLACache

        if isinstance(past_key_values, DeepSeekMLACache):
            start_pos = past_key_values.current_pos
            cache = past_key_values
        else:
            start_pos = 0
            cache = None

        seqlen = input_ids.size(1)
        freqs_cis = self.transformer.freqs_cis[start_pos : start_pos + seqlen]
        mask = self._causal_mask[:seqlen, :seqlen] if seqlen > 1 else None
        logits = self.transformer(
            tokens=input_ids, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask,
            cache=cache,
        )
        if cache is not None:
            cache.current_pos += seqlen
        return _CausalLMOutput(logits=logits.unsqueeze(1))


class ModelLoader(ForgeModel):
    """DeepSeek V3.2 model loader using the locally modified Transformer."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_V3_2_EXP: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-V3.2-Exp",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_V3_2_EXP

    # DeepSeek uses its own internal KV cache (attn.kv_cache / attn.pe_cache)
    # and ignores the HuggingFace StaticCache passed via past_key_values.
    # Setting this to False prevents llm_benchmark from marking the StaticCache
    # tensors as sharded — those sharded tensors would otherwise appear as
    # graph inputs with an unexpected layout to the SPMD partitioner, which
    # could corrupt the in_sharding seen by sdy.manual_computation and cause
    # the all_to_all_dispatch to hang.
    uses_external_kv_cache: bool = False

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
        #self.config = None

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
            "hidden_size":           "dim",
            "intermediate_size":     "inter_dim",
            "moe_intermediate_size": "moe_inter_dim",
            "num_hidden_layers":     "n_layers",
            "first_k_dense_replace": "n_dense_layers",
            "num_attention_heads":   "n_heads",
            "n_routed_experts":      "n_routed_experts",
            "n_shared_experts":      "n_shared_experts",
            "num_experts_per_tok":   "n_activated_experts",
            "n_group":               "n_expert_groups",
            "topk_group":            "n_limited_groups",
            "scoring_func":          "score_func",
            "routed_scaling_factor": "route_scale",
            "q_lora_rank":           "q_lora_rank",
            "kv_lora_rank":          "kv_lora_rank",
            "qk_nope_head_dim":      "qk_nope_head_dim",
            "qk_rope_head_dim":      "qk_rope_head_dim",
            "v_head_dim":            "v_head_dim",
            "vocab_size":            "vocab_size",
            "rope_theta":            "rope_theta",
            "index_n_heads":         "index_n_heads",
            "index_head_dim":        "index_head_dim",
            "index_topk":            "index_topk",
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
            #TODO (gengelage): Remove this.
            # config.json has n_dense_layers=3, which makes every layer dense
            # when num_layers is small (e.g. 2 for bringup tests).  Cap it to
            # n_layers-1 so there is always at least one MoE layer, matching
            # the ModelArgs default (n_dense_layers=1) used by the unit tests.
            n_dense = model_args_kwargs.get("n_dense_layers", 0)
            if n_dense >= self.num_layers:
                model_args_kwargs["n_dense_layers"] = max(0, self.num_layers - 1)
        model_args_kwargs.setdefault("max_batch_size", self.max_batch_size)
        args = ModelArgs(**model_args_kwargs)

        transformer = Transformer(args)

        if dtype_override is not None:
            transformer = transformer.to(dtype_override)

        # head is float32 in the original model (logits computed in fp32),
        # but .to(bfloat16) converts it — restore it so forward's .float() call
        # stays correct.
        transformer.head = transformer.head.to(torch.float32)

        # LayerNorm (used by k_norm in the Indexer) initialises its weight and
        # bias as float32 and explicitly calls x.float() in forward so that
        # F.layer_norm receives a float32 input.  .to(bfloat16) above silently
        # converts those parameters to bfloat16, causing a "mixed dtype" error
        # when the model is run on CPU — which the benchmark does to obtain
        # reference logits before running on the TT device.  Restoring them to
        # float32 here keeps the CPU execution path correct.
        for module in transformer.modules():
            if isinstance(module, LayerNorm):
                module.weight.data = module.weight.data.to(torch.float32)
                module.bias.data = module.bias.data.to(torch.float32)

        mesh_shape = (4, 8)  # Galaxy
        enable_sparse_mlp(transformer, mesh=mesh_shape, cluster_axis=0, config=args)

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

    def get_input_sharding(self):
        """Return sharding spec for input_ids on the Galaxy (4x8) mesh.

        DeepSeek's MoE layer uses all_to_all_dispatch which assumes that the
        batch dimension of the hidden states is split across the model axis
        (8 devices).  If input_ids are replicated the scatter indices inside
        sdy.manual_computation mismatch the sharded KV-cache, causing the
        all_to_all to deadlock on device.

        Sharding ("model", None) puts input_ids batch dim on the model axis
        (axis 1, size 8), giving batch/8 tokens per device — consistent with
        the KV-cache sharding ("model", None, None) and the passing unit test
        which uses shard_specs[args[0]] = ("_axis_1", None).

        Returns:
            Tuple ("model", None) for a [batch, seq] input_ids tensor.
        """
        return ("model", None)

    def load_shard_spec(self, model):
        """Build SPMD shard specifications for all model tensors.

        Sharding matches test_deepseek_v3_2_full_sparse_moe on a Galaxy (4, 8)
        mesh named ("batch", "model"):
          batch = axis 0, size 4  →  "_axis_0" in the unit test
          model = axis 1, size 8  →  "_axis_1" in the unit test

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
        from .modified_model import MLP

        shard_specs = {}
        t = model.transformer

        # ── RoPE / causal-mask buffers ────────────────────────────────────────
        # freqs_cis and _causal_mask are NOT sharded, but must be marked
        # as replicated in the 4×8 mesh so that any XLA sub-graph that slices
        # them (emitted eagerly during torch.compile tracing before the main
        # SPMD forward graph) is also compiled under the 4×8 mesh topology.
        # Without this, those slice sub-graphs compile under the default 1×32
        # mesh, producing tensors with an incompatible layout that causes a
        # segfault when the main 4×8 graph consumes them.
        shard_specs[t.freqs_cis] = (None, None, None)    # [max_seq, rope_hd, 2] — replicated
        shard_specs[model._causal_mask] = (None, None)   # [max_seq, max_seq] — replicated

        # ── Global tensors ────────────────────────────────────────────────────
        shard_specs[t.embed.weight] = (None, "batch")   # [vocab, dim]
        shard_specs[t.norm.weight] = ("batch",)         # [dim]
        shard_specs[t.head.weight] = (None, "batch")    # [vocab/world, dim]

        for layer in t.layers:
            attn = layer.attn

            # ── MLA attention — model-parallel on batch axis ──────────────────
            shard_specs[attn.wq_a.weight] = (None, "batch")    # [q_lora_rank, dim]
            shard_specs[attn.wkv_a.weight] = (None, "batch")   # [kv_lora_rank+rope_hd, dim]
            # shard_specs[attn.q_norm.weight] = (None,)
            # shard_specs[attn.kv_norm.weight] = (None,)
            shard_specs[attn.wq_b.weight] = ("batch", None)    # [n_heads*qk_hd, q_lora_rank]
            shard_specs[attn.wkv_b.weight] = ("batch", None)   # [n_heads*(nope+v_hd), kv_lora_rank]
            shard_specs[attn.wo.weight] = (None, "batch")      # [dim, n_heads*v_head_dim]
            # KV and positional caches — shard on model axis (batch dimension)
            shard_specs[attn.kv_cache] = ("model", None, None)
            shard_specs[attn.pe_cache] = ("model", None, None)

            # ── Indexer ───────────────────────────────────────────────────────
            if attn.indexer is not None:
                idx = attn.indexer
                shard_specs[idx.wq_b.weight] = ("batch", None)     # [n_idx_heads*idx_hd, q_lora_rank]
                shard_specs[idx.wk.weight] = (None, "batch")       # [idx_hd, dim]
                # shard_specs[idx.k_norm.weight] = (None,)
                # shard_specs[idx.k_norm.bias] = (None,)
                shard_specs[idx.weights_proj.weight] = (None, "batch")  # [n_idx_heads, dim]
                shard_specs[idx.k_cache] = ("model", None, None)

            # ── FFN ───────────────────────────────────────────────────────────
            ffn = layer.ffn
            if hasattr(ffn, "mlp"):
                # A2aSparseMLPWithSharedExperts (MoE layer)
                mlp = ffn.mlp
                shard_specs[mlp.router.gate.weight] = (None, "batch")
                shard_specs[mlp.experts.gate_proj] = (("batch", "model"), None, None)
                shard_specs[mlp.experts.up_proj] = (("batch", "model"), None, None)
                shard_specs[mlp.experts.down_proj] = (("batch", "model"), None, None)
                shard_specs[mlp.experts.gate_proj_bias] = (("batch", "model"), None)
                shard_specs[mlp.experts.up_proj_bias] = (("batch", "model"), None)
                shard_specs[mlp.experts.down_proj_bias] = (("batch", "model"), None)

                shared = getattr(ffn, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.w1.weight] = (None, "batch")
                    shard_specs[shared.w3.weight] = (None, "batch")
                    shard_specs[shared.w2.weight] = ("batch", None)
            else:
                # Dense MLP
                shard_specs[ffn.w1.weight] = ("model", "batch")
                shard_specs[ffn.w3.weight] = ("model", "batch")
                shard_specs[ffn.w2.weight] = ("batch", "model")

            # ── Layer norms ───────────────────────────────────────────────────
            shard_specs[layer.attn_norm.weight] = ("batch",)
            shard_specs[layer.ffn_norm.weight] = ("batch",)

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
