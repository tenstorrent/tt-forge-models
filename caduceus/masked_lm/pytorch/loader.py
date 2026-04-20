# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Caduceus model loader implementation for masked language modeling on DNA sequences.

The caduceus model requires mamba-ssm (CUDA-only) via trust_remote_code.
We install a pure PyTorch stub when the real package is unavailable.
"""
import math
import sys
import types
import importlib
import importlib.machinery

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_mamba_ssm_available():
    """Install a pure PyTorch stub for mamba_ssm if the real package is unavailable."""
    try:
        importlib.import_module("mamba_ssm.modules.mamba_simple")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
            self.bias = None

        def forward(self, x):
            dt = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return x.to(dt) * self.weight

    def _rms_norm_fn(
        x,
        weight,
        bias=None,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
    ):
        if residual is not None:
            x = x + residual
        residual_out = x
        if residual_in_fp32:
            residual_out = residual_out.float()
        dt = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = x.to(dt) * weight
        if bias is not None:
            x = x + bias
        if prenorm:
            return x, residual_out
        return x

    def _layer_norm_fn(
        x,
        weight,
        bias=None,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
    ):
        if residual is not None:
            x = x + residual
        residual_out = x
        if residual_in_fp32:
            residual_out = residual_out.float()
        x = F.layer_norm(x, weight.shape, weight, bias, eps)
        if prenorm:
            return x, residual_out
        return x

    class Mamba(nn.Module):
        def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
        ):
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = int(self.expand * self.d_model)
            self.dt_rank = (
                math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
            )
            self.use_fast_path = use_fast_path
            self.layer_idx = layer_idx

            self.in_proj = nn.Linear(
                self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
            )
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.activation = "silu"
            self.act = nn.SiLU()

            self.x_proj = nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )
            self.dt_proj = nn.Linear(
                self.dt_rank, self.d_inner, bias=True, **factory_kwargs
            )

            A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
            A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True

            self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
            self.D._no_weight_decay = True

            self.out_proj = nn.Linear(
                self.d_inner, self.d_model, bias=bias, **factory_kwargs
            )

        def forward(self, hidden_states, inference_params=None):
            dtype_in = hidden_states.dtype
            batch, seqlen, _ = hidden_states.shape

            xz = self.in_proj(hidden_states)
            x, z = xz.chunk(2, dim=-1)

            x_conv = x.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[..., :seqlen]
            x_conv = x_conv.transpose(1, 2)
            x_conv = self.act(x_conv)

            x_dbl = self.x_proj(x_conv)
            dt, B_param, C_param = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(self.dt_proj(dt))

            A = -torch.exp(self.A_log.float())
            x_f = x_conv.float()
            dt_f = dt.float()
            B_f = B_param.float()
            C_f = C_param.float()

            h = torch.zeros(
                batch,
                self.d_inner,
                self.d_state,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            ys = []
            for i in range(seqlen):
                dt_i = dt_f[:, i, :].unsqueeze(-1)
                B_i = B_f[:, i, :].unsqueeze(1)
                C_i = C_f[:, i, :].unsqueeze(1)
                x_i = x_f[:, i, :].unsqueeze(-1)
                h = torch.exp(dt_i * A) * h + dt_i * B_i * x_i
                ys.append((h * C_i).sum(dim=-1))
            y = torch.stack(ys, dim=1)

            y = y + x_f * self.D.float().unsqueeze(0).unsqueeze(0)
            y = y * F.silu(z.float())
            y = y.to(dtype_in)

            return self.out_proj(y)

    class Block(nn.Module):
        def __init__(
            self,
            dim,
            mixer_cls,
            mlp_cls=nn.Identity,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
        ):
            super().__init__()
            self.residual_in_fp32 = residual_in_fp32
            self.fused_add_norm = fused_add_norm
            self.mixer = mixer_cls(dim)
            self.norm = norm_cls(dim)

        def forward(self, hidden_states, residual=None, inference_params=None):
            if not self.fused_add_norm:
                residual = (
                    (hidden_states + residual)
                    if residual is not None
                    else hidden_states
                )
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                fused_fn = (
                    _rms_norm_fn if isinstance(self.norm, RMSNorm) else _layer_norm_fn
                )
                hidden_states, residual = fused_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
            return hidden_states, residual

    mamba_ssm = types.ModuleType("mamba_ssm")
    mamba_ssm.__version__ = "0.0.0"
    mamba_ssm.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", None)

    modules_pkg = types.ModuleType("mamba_ssm.modules")
    mamba_simple_mod = types.ModuleType("mamba_ssm.modules.mamba_simple")
    mamba_simple_mod.Mamba = Mamba
    mamba_simple_mod.Block = Block
    block_mod = types.ModuleType("mamba_ssm.modules.block")
    block_mod.Block = Block

    ops_pkg = types.ModuleType("mamba_ssm.ops")
    triton_pkg = types.ModuleType("mamba_ssm.ops.triton")
    layernorm_mod = types.ModuleType("mamba_ssm.ops.triton.layernorm")
    layernorm_mod.RMSNorm = RMSNorm
    layernorm_mod.layer_norm_fn = _layer_norm_fn
    layernorm_mod.rms_norm_fn = _rms_norm_fn
    layer_norm_mod = types.ModuleType("mamba_ssm.ops.triton.layer_norm")
    layer_norm_mod.RMSNorm = RMSNorm
    layer_norm_mod.layer_norm_fn = _layer_norm_fn
    layer_norm_mod.rms_norm_fn = _rms_norm_fn

    mamba_ssm.modules = modules_pkg
    modules_pkg.mamba_simple = mamba_simple_mod
    modules_pkg.block = block_mod
    mamba_ssm.ops = ops_pkg
    ops_pkg.triton = triton_pkg
    triton_pkg.layernorm = layernorm_mod
    triton_pkg.layer_norm = layer_norm_mod

    sys.modules["mamba_ssm"] = mamba_ssm
    sys.modules["mamba_ssm.modules"] = modules_pkg
    sys.modules["mamba_ssm.modules.mamba_simple"] = mamba_simple_mod
    sys.modules["mamba_ssm.modules.block"] = block_mod
    sys.modules["mamba_ssm.ops"] = ops_pkg
    sys.modules["mamba_ssm.ops.triton"] = triton_pkg
    sys.modules["mamba_ssm.ops.triton.layernorm"] = layernorm_mod
    sys.modules["mamba_ssm.ops.triton.layer_norm"] = layer_norm_mod


_ensure_mamba_ssm_available()

from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Optional

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


def _patch_caduceus_tie_weights(model_name):
    """Patch tie_weights in cached caduceus source for transformers 5.x compat.

    Transformers 5.x passes recompute_mapping= to tie_weights(), but the
    upstream caduceus code overrides tie_weights without **kwargs.  We rewrite
    the cached source file so the fix survives dynamic module reloads.
    """
    import os
    from huggingface_hub import try_to_load_from_cache

    for fname in ("modeling_caduceus.py", "modeling_rcps.py"):
        cached = try_to_load_from_cache(model_name, fname)
        if not cached or not os.path.isfile(cached):
            continue
        with open(cached, "r") as f:
            src = f.read()
        patched = src.replace(
            "def tie_weights(self):", "def tie_weights(self, **kwargs):"
        )
        patched = patched.replace(
            "super().tie_weights()", "super().tie_weights(**kwargs)"
        )
        if patched != src:
            with open(cached, "w") as f:
                f.write(patched)


class ModelVariant(StrEnum):
    """Available Caduceus model variants."""

    CADUCEUS_PH_131K = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"


class ModelLoader(ForgeModel):
    """Caduceus model loader implementation for masked language modeling on DNA sequences."""

    _VARIANTS = {
        ModelVariant.CADUCEUS_PH_131K: ModelConfig(
            pretrained_model_name="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CADUCEUS_PH_131K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Caduceus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        _patch_caduceus_tie_weights(self._variant_config.pretrained_model_name)

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Use a DNA sequence with [MASK] token for masked LM task
        masked_sequence = "ACCTGA[MASK]TTCTGAGTC"

        inputs = self.tokenizer(
            masked_sequence,
            return_tensors="pt",
            add_special_tokens=True,
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
