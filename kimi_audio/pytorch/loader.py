# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi-Audio model loader implementation for audio understanding and generation tasks.
"""
import importlib.machinery
import sys
import types
from typing import Optional
from unittest.mock import patch

import torch

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _ensure_flash_attn_stub():
    """Inject a minimal flash_attn stub so the model's remote code can import it.

    modeling_moonshot_kimia.py unconditionally requires flash_attn at module load time.
    This provides SDPA-based fallbacks so the model can run without the real flash_attn
    package (which requires CUDA to build).
    """
    if "flash_attn" in sys.modules:
        return

    def _flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        # flash-attn layout: (batch, seqlen, nheads, headdim)
        # The model casts float32 q/k/v to float16 before calling flash_attn (which
        # requires half precision).  Our SDPA stub doesn't need half precision, and
        # float16 SDPA on TT hardware produces NaN for this model.  Upgrade float16
        # inputs to bfloat16 so the computation runs in a dtype TT handles reliably.
        if q.dtype == torch.float16:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        q = q.transpose(1, 2)  # → (batch, nheads, seqlen, headdim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # GQA: expand k/v so head counts match q (flash_attn handles GQA natively).
        # Use unsqueeze+expand+reshape instead of repeat_interleave to avoid a
        # ttir.concat dimension-0 shape mismatch that repeat_interleave generates.
        if k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            B, nkH, S, D = k.shape
            k = k.unsqueeze(2).expand(B, nkH, n_rep, S, D).reshape(B, nkH * n_rep, S, D)
            v = v.unsqueeze(2).expand(B, nkH, n_rep, S, D).reshape(B, nkH * n_rep, S, D)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.transpose(1, 2)  # → (batch, seqlen, nheads, headdim)

    def _flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
    ):
        # Reconstruct padded tensors from packed layout, run SDPA, re-pack output.
        batch_size = cu_seqlens_q.shape[0] - 1
        nheads, headdim = q.shape[1], q.shape[2]
        q_p = torch.zeros(batch_size, max_seqlen_q, nheads, headdim, dtype=q.dtype, device=q.device)
        k_p = torch.zeros(batch_size, max_seqlen_k, nheads, headdim, dtype=k.dtype, device=k.device)
        v_p = torch.zeros(batch_size, max_seqlen_k, nheads, headdim, dtype=v.dtype, device=v.device)
        for b in range(batch_size):
            qs, qe = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
            ks, ke = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
            q_p[b, : qe - qs] = q[qs:qe]
            k_p[b, : ke - ks] = k[ks:ke]
            v_p[b, : ke - ks] = v[ks:ke]
        out_p = _flash_attn_func(q_p, k_p, v_p, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
        total_q = cu_seqlens_q[-1].item()
        out = torch.zeros(total_q, nheads, headdim, dtype=out_p.dtype, device=out_p.device)
        for b in range(batch_size):
            qs, qe = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
            out[qs:qe] = out_p[b, : qe - qs]
        return out

    def _index_first_axis(x, indices):
        return x[indices]

    def _pad_input(x, indices, batch, seqlen):
        output = torch.zeros((batch * seqlen,) + x.shape[1:], dtype=x.dtype, device=x.device)
        output[indices] = x
        return output.reshape(batch, seqlen, *x.shape[1:])

    def _unpad_input(hidden_states, attention_mask):
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        max_seqlen = seqlens.max().item()
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
        )
        b, s = hidden_states.shape[:2]
        return hidden_states.reshape(b * s, *hidden_states.shape[2:])[indices], indices, cu_seqlens, max_seqlen

    flash_attn_mod = types.ModuleType("flash_attn")
    flash_attn_mod.__version__ = "2.0.0"
    flash_attn_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    flash_attn_mod.flash_attn_func = _flash_attn_func
    flash_attn_mod.flash_attn_varlen_func = _flash_attn_varlen_func

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", None)
    bert_padding.index_first_axis = _index_first_axis
    bert_padding.pad_input = _pad_input
    bert_padding.unpad_input = _unpad_input

    flash_attn_mod.bert_padding = bert_padding
    sys.modules["flash_attn"] = flash_attn_mod
    sys.modules["flash_attn.bert_padding"] = bert_padding


_ensure_flash_attn_stub()


def _make_patched_apply_rotary_pos_emb():
    """Return a backward-compatible apply_rotary_pos_emb that accepts both the old
    (q, k, cos, sin, position_ids) and new (q, k, cos, sin, unsqueeze_dim=1) APIs.

    modeling_moonshot_kimia.py pre-indexes cos/sin with position_ids and then passes
    position_ids as the 5th positional argument. transformers 5.x renamed that argument
    to unsqueeze_dim (int), so unsqueeze(tensor) raises TypeError at runtime.
    """
    from transformers.models.qwen2.modeling_qwen2 import rotate_half

    def _compat_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim_or_pos_ids=1, **kwargs):
        # Old API passed a position_ids tensor; new API passes an int unsqueeze_dim.
        unsqueeze_dim = 1 if isinstance(unsqueeze_dim_or_pos_ids, torch.Tensor) else unsqueeze_dim_or_pos_ids
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    return _compat_apply_rotary_pos_emb


class ModelVariant(StrEnum):
    """Available Kimi-Audio model variants."""

    KIMI_AUDIO_7B_INSTRUCT = "7B_Instruct"


class ModelLoader(ForgeModel):
    """Kimi-Audio model loader implementation for audio understanding and generation tasks."""

    _VARIANTS = {
        ModelVariant.KIMI_AUDIO_7B_INSTRUCT: ModelConfig(
            pretrained_model_name="moonshotai/Kimi-Audio-7B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_AUDIO_7B_INSTRUCT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KimiAudio",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kimi-Audio model instance."""
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.dynamic_module_utils import get_imports
        import transformers.models.qwen2.modeling_qwen2 as _qwen2

        pretrained_model_name = self._variant_config.pretrained_model_name

        def fixed_get_imports(filename):
            imports = get_imports(filename)
            if not torch.cuda.is_available() and "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports

        patched_rope = _make_patched_apply_rotary_pos_emb()

        # is_flash_attn_2_available checks CUDA availability and always returns False on
        # non-CUDA systems, causing the remote model code to raise RuntimeError.  Patch
        # it to True during model loading so the import succeeds (the real flash_attn
        # calls are replaced by our SDPA stub injected above).
        # apply_rotary_pos_emb is patched for old-API compat (5th arg is position_ids
        # tensor in the remote code, but transformers 5.x expects int unsqueeze_dim).
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports), \
             patch("transformers.utils.is_flash_attn_2_available", return_value=True), \
             patch("transformers.utils.import_utils.is_flash_attn_2_available", return_value=True), \
             patch.object(_qwen2, "apply_rotary_pos_emb", patched_rope):

            # Pre-load the config so KimiAudioConfig is in sys.modules.  Transformers 5.x
            # moved rope_theta into rope_parameters dict; the remote model code still
            # accesses config.rope_theta directly.  Add a property before instantiation.
            config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
            config_cls = type(config)
            if not isinstance(config_cls.__dict__.get("rope_theta"), property):
                @property
                def _rope_theta_prop(self):
                    if isinstance(getattr(self, "rope_parameters", None), dict):
                        return self.rope_parameters.get("rope_theta", 10000.0)
                    return 10000.0
                config_cls.rope_theta = _rope_theta_prop

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                config=config,
                trust_remote_code=True,
                # Load in bfloat16 so the attention module's float32→float16 cast
                # (lines 338-348 of modeling_moonshot_kimia.py) is never triggered
                # (the cast only fires when input_dtype == torch.float32).  TT
                # hardware produces NaN for float16 operations, so avoiding the
                # cast is required for a valid TT run.
                torch_dtype=dtype_override if dtype_override is not None else torch.bfloat16,
                **kwargs,
            )
        model.eval()

        # The remote RotaryEmbedding._set_cos_sin_cache() calls emb.cos().to(dtype)
        # where dtype=bfloat16.  For some model configurations the cache computation
        # path leaves NaN values in cos_cached.  Re-compute every RotaryEmbedding
        # cache in float32 arithmetic (safe, cos(x) in [-1,1] never produces NaN),
        # then cast back to the original bfloat16 dtype so that CPU and TT both apply
        # RoPE in the same dtype and the PCC comparison stays tight.
        for mod in model.modules():
            if hasattr(mod, "_set_cos_sin_cache") and hasattr(mod, "cos_cached"):
                if mod.cos_cached is not None and mod.cos_cached.isnan().any():
                    orig_dtype = mod.cos_cached.dtype
                    mod._set_cos_sin_cache(
                        seq_len=mod.max_seq_len_cached,
                        device=mod.cos_cached.device,
                        dtype=torch.float32,
                    )
                    mod.cos_cached = mod.cos_cached.to(orig_dtype)
                    mod.sin_cached = mod.sin_cached.to(orig_dtype)

        self._model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Kimi-Audio model."""
        # The remote TikTokenTokenizer sets special-token attrs before calling
        # super().__init__(), which breaks transformers 5.x _special_tokens_map
        # guard.  We skip the tokenizer entirely and generate embeddings directly
        # via embed_tokens with dummy input IDs.
        # Attention_mask is intentionally omitted so padding_mask=None in the
        # attention layers, routing through flash_attn_func (causal=True) rather
        # than flash_attn_varlen_func.
        seq_len = 8
        # Use distinct token IDs so each position gets a different embedding.
        # All-zero IDs produce identical embeddings → zero-std outputs → NaN PCC.
        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            inputs_embeds = self._model.model.embed_tokens(input_ids)
        if dtype_override is not None:
            inputs_embeds = inputs_embeds.to(dtype_override)
        return {
            "inputs_embeds": inputs_embeds,
            "use_cache": False,
            "return_dict": False,
        }
