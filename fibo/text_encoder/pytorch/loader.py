# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO (briaai/FIBO) text-encoder loader.

FIBO is BRIA AI's 8B-parameter DiT text-to-image model. Its text encoder is
``SmolLM3-3B`` (``SmolLM3ForCausalLM``), shipped inside the gated ``briaai/FIBO``
repo under the ``text_encoder`` subfolder. FIBO conditions its DiT on the
transformer hidden states of this encoder (it discards the vocabulary head), so
this loader brings up the **base transformer only** (``SmolLM3Model`` via
``AutoModel``) and returns ``last_hidden_state``. Loading the base model rather
than the causal-LM head also avoids materializing the ``[batch, seq, 128256]``
vocabulary projection, which is essential at long context.

The DiT itself is brought up separately by the sibling ``fibo/pytorch`` loader.

Key facts (``text_encoder/config.json``):
  * ``SmolLM3`` decoder-only, 36 layers, hidden 2048, intermediate 11008
  * GQA: 16 query heads / 4 KV heads, head dim 128
  * ``max_position_embeddings = 65536`` (the model's max context length)
  * RoPE θ = 5e6 with NoPE (no positional encoding) on every 4th layer
  * tied embeddings, RMSNorm, SiLU MLP, no linear biases

This loader is tensor-parallel ready: ``get_mesh_config`` / ``load_shard_spec``
implement Megatron-1D (column→row) TP, the goal being TP-4 on a 4-chip device.

Reference: https://huggingface.co/briaai/FIBO
"""

import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# The gated FIBO repo bundles SmolLM3-3B under text_encoder/ and its tokenizer
# under tokenizer/. Accept the bria-fibo license and set HF_TOKEN before use.
_FIBO_REPO = "briaai/FIBO"
_TEXT_ENCODER_SUBFOLDER = "text_encoder"
_TOKENIZER_SUBFOLDER = "tokenizer"

# SmolLM3-3B max_position_embeddings — the architectural maximum context length
# the model supports. Any requested length is clamped down to this.
MAX_CONTEXT_LENGTH = 65536

# Default context length used when FIBO_TE_CONTEXT_LENGTH is unset (e.g. in CI).
# This is the largest context that compiles, runs, and passes PCC>=0.99 under
# TP-4 on a 4x Blackhole p300c (32 GB/chip) device: 32768 compiles but OOMs at
# runtime because the materialized [batch, 1, seq, seq] causal mask plus the
# O(seq^2) attention scores exceed per-chip DRAM. Override up to
# MAX_CONTEXT_LENGTH via FIBO_TE_CONTEXT_LENGTH (larger values need more chips
# or a mask-free attention path).
DEFAULT_CONTEXT_LENGTH = 24576
_CONTEXT_LENGTH_ENV = "FIBO_TE_CONTEXT_LENGTH"

# FIBO is trained on structured JSON captions; a stub caption gives realistic
# tokens for the bringup forward (it is tiled to fill the requested context).
_SAMPLE_PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)


class ModelVariant(StrEnum):
    """Available FIBO text-encoder variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """FIBO text-encoder (SmolLM3-3B) loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=_FIBO_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO-text-encoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            # The base SmolLM3 transformer produces the hidden-state embeddings
            # FIBO conditions on (no vocab head), so this is embedding generation.
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _context_length(self) -> int:
        """Resolve the context length, honoring FIBO_TE_CONTEXT_LENGTH."""
        raw = os.environ.get(_CONTEXT_LENGTH_ENV)
        if raw is None or raw.strip() == "":
            return DEFAULT_CONTEXT_LENGTH
        value = int(raw)
        if value < 1:
            value = 1
        return min(value, MAX_CONTEXT_LENGTH)

    def _load_tokenizer(self):
        """Load (and cache) the FIBO/SmolLM3 tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                _FIBO_REPO, subfolder=_TOKENIZER_SUBFOLDER
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the FIBO text encoder (base SmolLM3 transformer).

        Args:
            dtype_override: Optional ``torch.dtype`` for the weights (bf16 in the
                runner). If ``None``, the checkpoint dtype (bf16) is used.

        Returns:
            torch.nn.Module: ``SmolLM3Model`` in eval mode. Its forward returns a
            ``BaseModelOutputWithPast`` whose ``last_hidden_state`` is the
            conditioning embedding FIBO consumes.
        """
        model_kwargs = {
            "subfolder": _TEXT_ENCODER_SUBFOLDER,
            # SDPA is the device-friendly attention path; the mask is supplied by
            # load_inputs (see below) so the causal mask is materialized as a
            # standard bool mask rather than the int64 packed-sequence cumsum that
            # the mask-free path traces (unsupported by the TTNN accumulation kernel).
            "attn_implementation": "sdpa",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # AutoModel maps the SmolLM3ForCausalLM checkpoint onto the base
        # SmolLM3Model; the unused lm_head weights are dropped (expected).
        model = AutoModel.from_pretrained(_FIBO_REPO, **model_kwargs).eval()
        model.config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return sample inputs for the text encoder.

        Builds ``input_ids`` of shape ``[batch_size, context_length]`` by tiling a
        tokenized structured caption to fill the context. ``context_length``
        defaults to ``DEFAULT_CONTEXT_LENGTH`` (24576, the largest validated on
        TP-4) and is overridable up to the architectural max (65536) via
        ``FIBO_TE_CONTEXT_LENGTH``. An all-ones ``attention_mask`` is supplied so
        the model materializes a standard bool causal mask under torch.compile
        tracing; without it, the mask-free path traces an int64 packed-sequence
        cumsum that the TTNN accumulation kernel cannot compile.

        Args:
            dtype_override: Unused for the integer inputs (kept for the runner's
                calling convention).
            batch_size: Batch size (the task targets 1).

        Returns:
            dict: ``input_ids`` and ``attention_mask``, each
            ``LongTensor[batch_size, context_length]``.
        """
        self._load_tokenizer()
        ctx = self._context_length()

        base_ids = self.tokenizer(_SAMPLE_PROMPT, return_tensors="pt").input_ids[0]
        if base_ids.numel() == 0:
            fallback = self.tokenizer.bos_token_id or 0
            base_ids = torch.tensor([fallback], dtype=torch.long)

        reps = (ctx + base_ids.numel() - 1) // base_ids.numel()
        ids = base_ids.repeat(reps)[:ctx].unsqueeze(0).to(torch.long)
        input_ids = ids.repeat_interleave(batch_size, dim=0)
        attention_mask = torch.ones_like(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def load_config(self):
        """Load and return the SmolLM3 text-encoder config."""
        if self.config is None:
            self.config = AutoConfig.from_pretrained(
                _FIBO_REPO, subfolder=_TEXT_ENCODER_SUBFOLDER
            )
        return self.config

    def get_mesh_config(self, num_devices: int):
        """Return ``(mesh_shape, mesh_names)`` for tensor-parallel execution.

        Megatron-1D TP over a ``("batch", "model")`` mesh: the leading axis is
        size 1 (no data parallelism), the ``"model"`` axis is the TP degree. On a
        4-chip device this is a ``(1, 4)`` mesh = TP-4, the goal for this bringup.

        Args:
            num_devices: Total chip count (``xr.global_runtime_device_count()``).

        Returns:
            tuple: ``(mesh_shape, mesh_names)``.
        """
        if num_devices == 32:  # galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        cfg = self.load_config()
        model_axis = mesh_shape[1]
        assert (
            cfg.num_attention_heads % model_axis == 0
        ), "Query heads must be divisible by the model-axis size"
        assert (
            cfg.num_key_value_heads % model_axis == 0
        ), "KV heads must be divisible by the model-axis size (GQA)"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Return the Megatron-1D (column→row) shard spec for the encoder.

        Attention Q/K/V and MLP gate/up are column-parallel (output dim on the
        ``"model"`` axis); attention output and MLP down are row-parallel
        (contracting dim on ``"model"``), giving one all-reduce per attention and
        per MLP. Everything else (embeddings, RMSNorm weights) is replicated.
        GQA is preserved: with TP-4 each chip holds 4 query heads and 1 KV head.

        Args:
            model: the ``SmolLM3Model`` returned by ``load_model`` (on device).

        Returns:
            dict: ``{torch.nn.Parameter: partition_spec}``; params absent are
            replicated.
        """
        specs = {}
        for layer in model.layers:
            attn = layer.self_attn
            specs[attn.q_proj.weight] = ("model", "batch")
            specs[attn.k_proj.weight] = ("model", "batch")
            specs[attn.v_proj.weight] = ("model", "batch")
            specs[attn.o_proj.weight] = ("batch", "model")

            mlp = layer.mlp
            specs[mlp.gate_proj.weight] = ("model", "batch")
            specs[mlp.up_proj.weight] = ("model", "batch")
            specs[mlp.down_proj.weight] = ("batch", "model")
        return specs
