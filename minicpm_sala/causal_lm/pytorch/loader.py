# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-SALA model loader implementation for causal language modeling.
"""

import sys
import types
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_fla_for_tt_device():
    """
    Inject pure-PyTorch stubs for fla.ops submodules that normally require
    Triton GPU kernels. MiniCPM-SALA's remote code imports fla ops at module
    level; these stubs satisfy the import and provide reference implementations
    that torch.compile can trace without a CUDA/Triton GPU.
    """

    def _naive_chunk_simple_gla(
        q, k, v, g, initial_state=None, output_final_state=False, chunk_size=64, scale=None
    ):
        q, k, v, g = [rearrange(x, "b t h ... -> b h t ...").to(torch.float32) for x in [q, k, v, g]]
        if scale is None:
            scale = 1.0 / q.shape[-1] ** 0.5
        T = q.shape[-2]
        BT = chunk_size
        pad_len = (BT - (T % BT)) % BT
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            g = F.pad(g, (0, pad_len))
        decay = g
        B, H, T1, K = q.shape
        V = v.shape[-1]
        q = q * scale
        q, k, v, decay = [
            rearrange(x, "b h (n c) d -> b h n c d", c=BT)
            for x in [q, k, v, decay.unsqueeze(-1)]
        ]
        decay = decay.squeeze(-1).cumsum(-1)
        L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
        S = k.new_zeros(B, H, K, V)
        if initial_state is not None:
            S = initial_state.to(torch.float32)
        chunk_outputs = []
        for i in range(T1 // BT):
            q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
            attn = q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]
            o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
            chunk_outputs.append(o_inter + attn @ v_i)
            S = S * decay[:, :, i, -1, None, None].exp() + (
                k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
            ).transpose(-1, -2) @ v_i
        if not output_final_state:
            S = None
        o = torch.stack(chunk_outputs, dim=2)  # [B, H, n, c, V]
        o = rearrange(o, "b h n c d -> b (n c) h d")[:, :T]
        return o, S

    def _naive_recurrent_simple_gla(q, k, v, g, scale=None, initial_state=None, output_final_state=True):
        dtype = q.dtype
        q, k, v, g = [x.transpose(1, 2).to(torch.float32) for x in (q, k, v, g)]
        B, H, T, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K**-0.5
        q = q * scale
        S = q.new_zeros(B, H, K, V)
        if initial_state is not None:
            S = S + initial_state.to(torch.float32)
        step_outputs = []
        for i in range(T):
            gate = g[:, :, i].exp()
            kv = k[:, :, i].unsqueeze(-1) * v[:, :, i].unsqueeze(-2)
            S = S * gate.unsqueeze(-1).unsqueeze(-1) + kv
            step_outputs.append((q[:, :, i].unsqueeze(-1) * S).sum(-2))
        if not output_final_state:
            S = None
        o = torch.stack(step_outputs, dim=2)  # [B, H, T, V]
        return o.transpose(1, 2).to(dtype), S

    def _chunk_simple_gla(
        q, k, v, g=None, g_gamma=None, scale=None,
        initial_state=None, output_final_state=False,
        cu_seqlens=None, cu_seqlens_cpu=None, **kwargs,
    ):
        if g is None and g_gamma is not None:
            B, T, H, _ = q.shape
            g = g_gamma.unsqueeze(0).unsqueeze(0).expand(B, T, H)
        return _naive_chunk_simple_gla(q, k, v, g, initial_state, output_final_state, scale=scale)

    def _fused_recurrent_simple_gla(
        q, k, v, g=None, g_gamma=None, scale=None,
        initial_state=None, output_final_state=False,
        reverse=False, cu_seqlens=None, **kwargs,
    ):
        if g is None and g_gamma is not None:
            B, T, H, _ = q.shape
            g = g_gamma.unsqueeze(0).unsqueeze(0).expand(B, T, H)
        return _naive_recurrent_simple_gla(q, k, v, g, scale, initial_state, output_final_state)

    def _prepare_lens_from_mask(mask):
        return mask.sum(dim=-1, dtype=torch.int32)

    def _prepare_cu_seqlens_from_mask(mask, dtype=torch.int32):
        lens = _prepare_lens_from_mask(mask)
        return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))

    def _tensor_cache(fn):
        return fn

    # Inject fla.utils stub (provides tensor_cache)
    if "fla.utils" not in sys.modules:
        m_utils = types.ModuleType("fla.utils")
        sys.modules["fla.utils"] = m_utils
    sys.modules["fla.utils"].tensor_cache = _tensor_cache

    # Block fla.ops/__init__.py from running (it imports triton eagerly)
    for name in ("fla.ops", "fla.ops.utils", "fla.ops.simple_gla"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # fla.ops.utils.index stubs
    m_idx = types.ModuleType("fla.ops.utils.index")
    m_idx.prepare_lens_from_mask = _prepare_lens_from_mask
    m_idx.prepare_cu_seqlens_from_mask = _prepare_cu_seqlens_from_mask
    sys.modules["fla.ops.utils.index"] = m_idx

    # fla.ops.simple_gla.chunk_simple_gla stub
    sys.modules["fla.ops.simple_gla"].chunk_simple_gla = _chunk_simple_gla

    # fla.ops.simple_gla.fused_recurrent stub
    m_fr = types.ModuleType("fla.ops.simple_gla.fused_recurrent")
    m_fr.fused_recurrent_simple_gla = _fused_recurrent_simple_gla
    sys.modules["fla.ops.simple_gla.fused_recurrent"] = m_fr


def _patch_model_gla_attention(model):
    """
    Patch all SALA/Lightning GLA attention modules in the loaded model so that
    a 4D SDPA causal attention mask ([B,1,T,T]) is replaced with None before
    it reaches attn_fn.  The 4D mask is produced by
    _prepare_4d_causal_attention_mask_for_sdpa when running with torch.compile;
    Simple GLA is inherently causal (lower-triangular decay mask), so the
    padding mask is only needed for multi-sequence packing.  For batch_size=1
    inference (no padding), passing None is semantically equivalent.
    """
    for module in model.modules():
        if not hasattr(module, "attn_fn"):
            continue
        orig = module.attn_fn

        def _make_wrapper(orig_fn):
            def _wrapped(*args, **kwargs):
                mask = kwargs.get("attention_mask")
                if mask is not None and mask.dim() == 4:
                    kwargs["attention_mask"] = None
                return orig_fn(*args, **kwargs)
            return _wrapped

        module.attn_fn = _make_wrapper(orig)


class ModelVariant(StrEnum):
    """Available MiniCPM-SALA model variants."""

    MINICPM_SALA_9B = "9B"


class ModelLoader(ForgeModel):
    """MiniCPM-SALA model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINICPM_SALA_9B: LLMModelConfig(
            pretrained_model_name="openbmb/MiniCPM-SALA",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINICPM_SALA_9B

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MiniCPM-SALA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _patch_fla_for_tt_device()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if (
            hasattr(config, "rope_scaling")
            and isinstance(config.rope_scaling, dict)
            and config.rope_scaling.get("rope_type") == "default"
        ):
            config.rope_scaling = None
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        ).eval()

        _patch_model_gla_attention(model)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
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

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
