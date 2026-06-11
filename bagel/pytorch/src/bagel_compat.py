# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims + clean-forward wrapper for ByteDance BAGEL (JiaxinGe/Diffusers-BAGEL).

BAGEL ships as a `trust_remote_code` diffusers ``BagelPipeline`` whose custom ``pipeline.py``:
  * hard-imports ``flash_attn`` (CUDA-only) at module top,
  * targets transformers ~4.4x (incompatible with newer transformers loading paths),
  * builds the LM under ``init_empty_weights()`` so the rotary ``inv_freq`` (a non-persistent
    buffer, absent from the checkpoint) is left on the meta device and materialises as NaN.

This module isolates every workaround needed to import the remote code, load the
``bagel_model`` component, and run its Qwen2 Mixture-of-Transformers (MoT) backbone through a
clean, fixed-shape, tensors-only forward (the text / "und" understanding path) that lowers to
StableHLO/TTNN. The image-generation ("gen") expert path is intentionally not exercised here.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from typing import Optional

import torch
from huggingface_hub import snapshot_download

_HF_REPO = "JiaxinGe/Diffusers-BAGEL"
_REMOTE_MODULE_NAME = "bagel_pipeline_remote"


# --------------------------------------------------------------------------------------------
# Environment shims
# --------------------------------------------------------------------------------------------
def _install_flash_attn_stub() -> None:
    """flash_attn is CUDA-only and uninstallable here; the bringup uses the sdpa path instead.

    The remote ``pipeline.py`` does ``from flash_attn import flash_attn_varlen_func`` at module
    top, and diffusers calls ``importlib.util.find_spec("flash_attn")`` (needs a real __spec__).
    """
    if "flash_attn" in sys.modules and getattr(
        sys.modules["flash_attn"], "_tt_stub", False
    ):
        return

    fa = types.ModuleType("flash_attn")
    fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    fa.__version__ = "2.9.9"
    fa._tt_stub = True

    def _no_flash(*_a, **_k):
        raise NotImplementedError(
            "flash_attn is stubbed for TT bringup; the clean forward uses the sdpa path."
        )

    fa.flash_attn_varlen_func = _no_flash
    sys.modules["flash_attn"] = fa
    for sub in ("flash_attn.bert_padding", "flash_attn.flash_attn_interface"):
        m = types.ModuleType(sub)
        m.__spec__ = importlib.machinery.ModuleSpec(sub, loader=None)
        sys.modules[sub] = m


def _install_rope_default_shim() -> None:
    """transformers >=5.x dropped ROPE_INIT_FUNCTIONS["default"]; register standard RoPE."""
    import transformers.modeling_rope_utils as _rope

    if "default" in _rope.ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0

    _rope.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def load_remote_module():
    """Import the BAGEL remote ``pipeline.py`` with all shims applied. Returns the module."""
    _install_flash_attn_stub()
    _install_rope_default_shim()

    if _REMOTE_MODULE_NAME in sys.modules:
        return sys.modules[_REMOTE_MODULE_NAME]

    snapshot = snapshot_download(_HF_REPO)
    pipeline_py = os.path.join(snapshot, "pipeline.py")
    spec = importlib.util.spec_from_file_location(_REMOTE_MODULE_NAME, pipeline_py)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: a @dataclass in pipeline.py introspects sys.modules[cls.__module__].
    sys.modules[_REMOTE_MODULE_NAME] = mod
    spec.loader.exec_module(mod)

    # The forced CUDA-only sdpa backend would error on CPU/XLA; let the default kernel run.
    mod.sdpa_kernel = lambda *_a, **_k: contextlib.nullcontext()

    # transformers >=5.x loading-path shims (BAGEL's Bagel class predates them).
    Bagel = mod.Bagel
    if (
        not hasattr(Bagel, "all_tied_weights_keys")
        or Bagel.all_tied_weights_keys is None
    ):
        Bagel.all_tied_weights_keys = {}

    def _init_weights_noop(self, *_a, **_k):
        # BAGEL's _init_weights is a manual hook; transformers >=5.x auto-calls it with a
        # module arg during loading. No-op preserves the loaded checkpoint weights.
        return None

    for cls_name in ("Bagel", "Qwen2PreTrainedModel", "SiglipPreTrainedModel"):
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            cls._init_weights = _init_weights_noop

    return mod


def _repair_rotary_inv_freq(mod, model) -> int:
    """Recompute every Qwen2RotaryEmbedding.inv_freq on CPU.

    BAGEL builds the LM under ``init_empty_weights()`` (meta device); ``inv_freq`` is a
    non-persistent buffer (not in the checkpoint), so it stays NaN after materialisation.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, mod.Qwen2RotaryEmbedding):
            inv_freq, scaling = m.rope_init_fn(m.config, device=torch.device("cpu"))
            m.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)
            m.original_inv_freq = m.inv_freq
            m.attention_scaling = scaling
            n += 1
    return n


# --------------------------------------------------------------------------------------------
# Clean text-backbone wrapper
# --------------------------------------------------------------------------------------------
class BagelTextBackbone(torch.nn.Module):
    """Tensors-only forward over the full Qwen2 MoT backbone (text / "und" path).

    forward(input_ids: LongTensor[B, S] | [S]) -> hidden_states[S, hidden].

    Drives ``Qwen2Model.forward_train`` with a single padded sequence and a List causal mask so
    attention takes the sdpa branch (not flex_attention / flash_attn). All tokens route to the
    understanding ("und") expert; the generation ("gen") expert is unused.
    """

    def __init__(self, bagel_model):
        super().__init__()
        self.bagel = bagel_model
        self.backbone = bagel_model.language_model.model  # Qwen2Model (MoT)
        # decoder_layer.forward dispatches on self.training: train -> forward_train (sdpa);
        # eval -> forward_inference (flash_attn + KV cache). Config dropout=0.0, so train mode
        # is numerically eval-equivalent here.
        self.backbone.train()

    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        ids = input_ids.reshape(-1).to(torch.long)
        S = ids.shape[0]
        emb = self.backbone.embed_tokens(ids)  # (S, hidden)
        position_ids = torch.arange(S, dtype=torch.long, device=ids.device)
        cos, sin = self.backbone.rotary_emb(emb, position_ids.unsqueeze(0))
        pe = (cos.squeeze(0), sin.squeeze(0))
        # additive causal mask; bf16-safe finite min (literal -inf can yield NaN in bf16 softmax)
        neg = torch.finfo(torch.bfloat16).min
        causal = torch.triu(
            torch.full((S, S), neg, device=ids.device), diagonal=1
        ).unsqueeze(0)
        und = torch.arange(S, dtype=torch.long, device=ids.device)
        gen = torch.empty(0, dtype=torch.long, device=ids.device)
        for layer in self.backbone.layers:
            emb = layer.forward_train(
                packed_sequence=emb,
                sample_lens=[S],
                attention_mask=[causal],
                packed_position_embeddings=pe,
                packed_und_token_indexes=und,
                packed_gen_token_indexes=gen,
            )
        return self.backbone.norm(emb)


def load_bagel_text_backbone(dtype: torch.dtype = torch.bfloat16) -> BagelTextBackbone:
    """Load the bagel_model component and wrap its MoT backbone for a clean forward."""
    mod = load_remote_module()
    snapshot = snapshot_download(_HF_REPO)
    model = mod.Bagel.from_pretrained(
        os.path.join(snapshot, "bagel_model"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _repair_rotary_inv_freq(mod, model)
    return BagelTextBackbone(model)
