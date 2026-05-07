# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaDA2.0-mini model loader.

LLaDA2.0 (Large Language Diffusion model with mAsking) is a discrete diffusion LM
from inclusionAI. It registers in HuggingFace as ``AutoModelForCausalLM`` via
``auto_map`` so it slots into the standard CausalLM API, but the model is *not*
trained with the next-token objective - it predicts masked tokens with full
bidirectional context, BERT-style, applied iteratively at sample time.

Three consequences for the bring-up loader:

1. The model code at ``modeling_llada2_moe.py:866-876`` requires the caller to
   supply an explicit ``(batch_size, 1, seq_length, seq_length)`` block
   attention mask - it refuses ``None`` and the usual ``(B, T)`` HF mask. We
   build that mask here in :meth:`ModelLoader.load_inputs`.
2. Generation is iterative mask-denoising rather than autoregressive next-token
   sampling. This loader exposes only the forward pass; producing real text
   requires a diffusion sampling loop, which is out of scope for compiler
   bring-up.
3. The upstream MoE inference path (``LLaDA2MoeSparseMoeBlock.moe_infer``) uses
   ``.cpu().numpy()``, ``.item()`` and a Python loop with a data-dependent
   ``continue`` to dispatch tokens to experts. That is a graph-break gauntlet
   under ``torch.compile`` and will not lower through TT-MLIR. When
   ``compile_friendly_moe=True`` (the default) :meth:`ModelLoader.load_model`
   patches each MoE block's ``forward`` with a mathematically-equivalent
   formulation that avoids data-dependent control flow. See
   :func:`_install_compile_friendly_moe` for the rewrite and
   ``tests/torch/models/llada2_mini/test_llada2_mini.py`` for the CPU-side
   equivalence check.

The loader follows the ``ForgeModel`` interface used by the rest of
``tt_forge_models``. By default it instantiates the model from config with
random weights (no 32 GB checkpoint download), which is what we want for
"does the graph compile and produce numerically equivalent logits" tests.
Pass ``pretrained=True`` to :meth:`load_model` to switch to real weights.
"""
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _compile_friendly_moe_forward(self, hidden_states):
    """Drop-in replacement for ``LLaDA2MoeSparseMoeBlock.forward``.

    Functionally equivalent to the eager (``self.training=False``) path of the
    upstream block, but expressed without data-dependent control flow so that
    ``torch.compile`` can trace it as a single graph and TT-MLIR can lower it.

    Upstream ``moe_infer`` does:

        for each expert i in [0, E):
            collect tokens routed to expert i        (.cpu(), .item(), if==0)
            run expert i on those tokens
            scatter outputs back, weighted by topk_weight

    This rewrite computes:

        routing_weights[b, e] = sum_k topk_weight[b, k] * 1[topk_idx[b, k] == e]
        out[b, h]             = sum_e routing_weights[b, e] * expert_e(x[b])[h]

    which is mathematically the same weighted-sum-of-top-k-experts but uses
    only standard tensor ops. Iteration over ``self.experts`` is a Python-level
    ``ModuleList`` traversal that ``torch.compile`` unrolls at trace time, so
    no graph break occurs.

    Trade-offs:
      * Every expert runs on every token (vs. only top-k tokens upstream),
        increasing FLOPs by ``num_experts / num_experts_per_tok`` (32x for the
        LLaDA2.0-mini config). This is acceptable for forward-parity bring-up
        tests on small ``seq_len``; perf is not the goal here.
      * Tracing unrolls the 256-expert ``ModuleList`` into ~256 expert
        sub-graphs, so first-time compile is slow.
    """
    identity = hidden_states
    bsz, seq_len, h = hidden_states.shape

    topk_idx, topk_weight, router_logits = self.gate(hidden_states)
    flat_x = hidden_states.view(-1, h)

    num_experts = len(self.experts)
    expert_range = torch.arange(num_experts, device=flat_x.device)
    one_hot = (topk_idx.unsqueeze(-1) == expert_range).to(topk_weight.dtype)
    routing_weights = (topk_weight.unsqueeze(-1) * one_hot).sum(dim=1)

    accum = torch.zeros(
        flat_x.shape[0], h, dtype=topk_weight.dtype, device=flat_x.device
    )
    for i, expert in enumerate(self.experts):
        weight_i = routing_weights[:, i : i + 1]
        expert_out = expert(flat_x).to(topk_weight.dtype)
        accum = accum + expert_out * weight_i
    y = accum.to(flat_x.dtype).view(bsz, seq_len, h)

    if self.config.num_shared_experts is not None:
        y = y + self.shared_experts(identity)

    return y, (
        router_logits.view(bsz, seq_len, -1),
        topk_idx.view(bsz, seq_len, -1),
    )


def _install_compile_friendly_moe(model: nn.Module) -> int:
    """Patch every ``LLaDA2MoeSparseMoeBlock`` in ``model`` to use the
    compile-friendly forward defined above.

    Returns the number of MoE blocks patched (0 when ``num_layers <
    first_k_dense_replace``, in which case there are no MoE layers and the
    patch is a no-op).
    """
    patched = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if type(mlp).__name__ != "LLaDA2MoeSparseMoeBlock":
            continue
        mlp.forward = _compile_friendly_moe_forward.__get__(mlp, type(mlp))
        patched += 1
    return patched


class ModelVariant(StrEnum):
    """Available LLaDA2.0 model variants."""

    LLADA_2_0_MINI = "LLaDA2.0-mini"


class ModelLoader(ForgeModel):
    """LLaDA2.0-mini loader for compiler bring-up on Tenstorrent hardware."""

    _VARIANTS = {
        ModelVariant.LLADA_2_0_MINI: LLMModelConfig(
            pretrained_model_name="inclusionAI/LLaDA2.0-mini",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLADA_2_0_MINI

    # Synthetic input defaults for bring-up (kept small so first compile is fast).
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_SEQ_LEN = 8

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        num_layers: Optional[int] = 1,
    ):
        """Initialize the LLaDA2.0 loader.

        Args:
            variant: Which model variant to use. Defaults to ``LLADA_2_0_MINI``.
            num_layers: Number of transformer layers to instantiate. Defaults to
                1 for fast first-time bring-up; pass ``None`` to use the full
                config depth (20 layers for the mini variant).
        """
        super().__init__(variant)
        self.num_layers = num_layers
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return dashboard / reporting metadata for this model variant."""
        return ModelInfo(
            model="LLaDA2.0",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,  # registers as CausalLM via auto_map
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the LLaDA tokenizer (custom ``PreTrainedTokenizerFast`` with
        ``<|mask|>`` / ``[gMASK]`` special tokens)."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_config(self, num_layers: Optional[int] = None):
        """Load and return the model config with bring-up overrides applied.

        Overrides:
          - ``num_hidden_layers`` -> ``self.num_layers`` (or argument override)
          - ``use_cache = False``: avoids ``DynamicCache`` instantiation in the
            traced graph; also semantically irrelevant for diffusion LMs.
          - ``_attn_implementation = "eager"``: SDPA-fused attention does not
            cleanly lower through TT-MLIR yet.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        layers = num_layers if num_layers is not None else self.num_layers
        if layers is not None:
            config.num_hidden_layers = layers
        config.use_cache = False
        config._attn_implementation = "eager"

        self.config = config
        return config

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = torch.bfloat16,
        pretrained: bool = False,
        seed: int = 0,
        compile_friendly_moe: bool = True,
        **kwargs,
    ):
        """Instantiate and return the LLaDA2.0 model.

        Args:
            dtype_override: dtype to cast the model to after construction.
                Defaults to ``torch.bfloat16`` to match the rest of the TT
                bring-up suite.
            pretrained: If ``True``, download the full HF checkpoint and
                load weights via ``from_pretrained``. Defaults to ``False``
                (random init from config) to keep the test cheap and offline.
                When ``True``, ``num_layers`` is still respected if set.
            seed: RNG seed for random-init reproducibility.
            compile_friendly_moe: If ``True`` (default), replace every
                ``LLaDA2MoeSparseMoeBlock.forward`` with a graph-breakless
                equivalent (see :func:`_compile_friendly_moe_forward`). Required
                for ``torch.compile(backend="tt")`` to lower MoE layers; the
                upstream ``moe_infer`` path is dynamo-hostile. Set to ``False``
                when comparing against the unmodified eager forward (e.g. in
                the CPU-side equivalence test).

        Returns:
            torch.nn.Module: LLaDA2.0 model in eval mode.
        """
        config = self.load_config()
        if self.tokenizer is None:
            self._load_tokenizer()

        torch.manual_seed(seed)
        if pretrained:
            pretrained_model_name = self._variant_config.pretrained_model_name
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                config=config,
                **kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, **kwargs
            )

        if dtype_override is not None:
            model = model.to(dtype_override)
        model = model.eval()

        if compile_friendly_moe:
            _install_compile_friendly_moe(model)

        self.model = model
        return model

    def load_inputs(
        self,
        *,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        seed: int = 0,
    ):
        """Return synthetic inputs for the LLaDA2.0 forward pass.

        Returns a dict ready to splat into ``model(**inputs)`` containing:

          - ``input_ids``: ``(B, T) int64`` random token ids in ``[0, vocab_size)``.
          - ``attention_mask``: ``(B, 1, T, T) bool``, all True. LLaDA's
            modeling code requires this exact shape and refuses ``None`` or
            the standard ``(B, T)`` mask. All-True is the right semantic for
            a diffusion LM (fully bidirectional attention).
        """
        if self.config is None:
            self.load_config()

        B = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
        T = seq_len if seq_len is not None else self.DEFAULT_SEQ_LEN

        gen = torch.Generator().manual_seed(seed)
        input_ids = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(B, T),
            dtype=torch.long,
            generator=gen,
        )
        attention_mask = torch.ones(B, 1, T, T, dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
