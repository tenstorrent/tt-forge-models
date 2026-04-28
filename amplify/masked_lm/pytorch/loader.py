# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AMPLIFY model loader implementation for masked language modeling on protein sequences.
"""
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional


def _ensure_xformers_stub():
    """Inject a minimal xformers stub so the AMPLIFY hub code can import without CUDA xformers.

    The AMPLIFY remote model code uses xformers.ops.SwiGLU (weight layout: w12 + w3) and
    xformers.ops.memory_efficient_attention (only invoked on CUDA; CPU path uses
    scaled_dot_product_attention). We provide compatible pure-PyTorch replacements.
    """
    if "xformers" in sys.modules:
        return

    class SwiGLU(nn.Module):
        def __init__(
            self, in_features, hidden_features, out_features=None, bias=True, **kwargs
        ):
            super().__init__()
            out_features = out_features or in_features
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x):
            x1, x2 = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, **kwargs):
        # Transpose from (B, M, H, K) to (B, H, M, K) for sdpa
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p)
        return out.transpose(1, 2)

    ops_mod = types.ModuleType("xformers.ops")
    ops_mod.SwiGLU = SwiGLU
    ops_mod.memory_efficient_attention = memory_efficient_attention

    xformers_mod = types.ModuleType("xformers")
    xformers_mod.ops = ops_mod

    sys.modules["xformers"] = xformers_mod
    sys.modules["xformers.ops"] = ops_mod


from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available AMPLIFY model variants."""

    AMPLIFY_350M = "chandar-lab/AMPLIFY_350M"


class ModelLoader(ForgeModel):
    """AMPLIFY model loader implementation for masked language modeling on protein sequences."""

    _VARIANTS = {
        ModelVariant.AMPLIFY_350M: ModelConfig(
            pretrained_model_name="chandar-lab/AMPLIFY_350M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AMPLIFY_350M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AMPLIFY",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        _ensure_xformers_stub()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_xformers_stub()
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        # Transformers 5.x lazy loading leaves non-buffer tensor attributes as meta
        # tensors. freqs_cis is not registered via register_buffer so it is not in
        # the state_dict and never gets materialized; recompute it from config.
        if hasattr(model, "freqs_cis") and model.freqs_cis.is_meta:
            cfg = model.config
            dim = cfg.hidden_size // cfg.num_attention_heads
            freqs = 1.0 / (
                10000.0 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
            t = torch.arange(cfg.max_length)
            freqs = torch.outer(t, freqs).float()
            model.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # The hub's _att_block calls attn.view() after sdpa().transpose(1,2).
        # sdpa().transpose() is non-contiguous and cannot be directly viewed;
        # xformers always returned contiguous outputs, masking this bug.
        # Patch the class to add .contiguous() before .view().
        _block_cls = type(model.transformer_encoder[0])
        _hub_mod = sys.modules[_block_cls.__module__]
        _apply_rotary_emb = _hub_mod.apply_rotary_emb
        _mem_eff_attn = _hub_mod.memory_efficient_attention

        def _att_block_patched(
            self, x, attention_mask, freqs_cis, output_attentions
        ):
            batch_size, seq_len, _ = x.shape
            xq, xk, xv = self.q(x), self.k(x), self.v(x)
            xq = xq.view(
                batch_size, seq_len, self.config.num_attention_heads, self.d_head
            )
            xk = xk.view(
                batch_size, seq_len, self.config.num_attention_heads, self.d_head
            )
            xv = xv.view(
                batch_size, seq_len, self.config.num_attention_heads, self.d_head
            )
            xq, xk = _apply_rotary_emb(xq, xk, freqs_cis)
            attn_weights = None
            if output_attentions:
                attn_weights = (
                    xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1)
                ) / (xq.size(-1) ** 0.5)
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = attn_weights.softmax(-1)
            if x.is_cuda:
                attn = _mem_eff_attn(
                    query=xq,
                    key=xk,
                    value=xv,
                    attn_bias=attention_mask,
                    p=self.config.dropout_prob if self.training else 0,
                )
            else:
                attn = F.scaled_dot_product_attention(
                    query=xq.transpose(1, 2),
                    key=xk.transpose(1, 2),
                    value=xv.transpose(1, 2),
                    attn_mask=attention_mask,
                    dropout_p=self.config.dropout_prob if self.training else 0,
                ).transpose(1, 2).contiguous()
            attn_scores = self.wo(
                attn.view(
                    batch_size,
                    seq_len,
                    self.config.num_attention_heads * self.d_head,
                )
            )
            return (self.resid_dropout(attn_scores), attn_weights)

        _block_cls._att_block = _att_block_patched

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Short protein sequence with a mask token for masked LM task
        masked_sequence = "MGSSHHHHHHSSGLVPRGSHM<mask>GSSHHHHHHSSGLVPRGSHM"

        inputs = self.tokenizer(
            masked_sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # AMPLIFY expects an additive attention mask (0.0 = attend, -inf = mask)
        inputs["attention_mask"] = torch.where(
            inputs["attention_mask"].bool(),
            torch.tensor(0.0),
            torch.tensor(float("-inf")),
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens
