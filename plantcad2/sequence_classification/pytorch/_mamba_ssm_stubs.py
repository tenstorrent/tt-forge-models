# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stub implementations for mamba_ssm package required by PlantCAD2/Caduceus models.

mamba_ssm requires CUDA to build from source. These stubs provide the minimal
interface needed to instantiate and trace Caduceus models for compilation.
"""
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install():
    # Return early only if our complete stub is already installed
    if "mamba_ssm.models.config_mamba" in sys.modules:
        return

    # --- leaf modules ---

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
            super().__init__()
            self.eps = eps
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))

        @property
        def bias(self):
            return None

        def forward(self, x):
            return F.rms_norm(x, self.weight.shape, self.weight, self.eps)

    def layer_norm_fn(
        x,
        weight,
        bias,
        residual=None,
        eps=1e-5,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        if residual is not None:
            x = x + residual
        residual_out = x.float() if residual_in_fp32 else x
        if is_rms_norm:
            x_normed = F.rms_norm(x, weight.shape, weight, eps)
        else:
            x_normed = F.layer_norm(x, weight.shape, weight, bias, eps)
        if prenorm:
            return x_normed, residual_out
        return x_normed

    def rms_norm_fn(
        x,
        weight,
        bias,
        residual=None,
        eps=1e-5,
        prenorm=False,
        residual_in_fp32=False,
    ):
        return layer_norm_fn(
            x,
            weight,
            bias,
            residual=residual,
            eps=eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
        )

    class MambaConfig:
        def __init__(self, d_model=2560, n_layer=64, vocab_size=50277, **kwargs):
            self.d_model = d_model
            self.n_layer = n_layer
            self.vocab_size = vocab_size
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Mamba(nn.Module):
        def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            layer_idx=None,
            device=None,
            dtype=None,
            **kwargs
        ):
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            d_inner = int(expand * d_model)
            self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False, **factory_kwargs)
            self.out_proj = nn.Linear(d_inner, d_model, bias=False, **factory_kwargs)

        def forward(self, hidden_states, inference_params=None, **kwargs):
            xz = self.in_proj(hidden_states)
            x, z = xz.chunk(2, dim=-1)
            return self.out_proj(x * F.silu(z))

    class Mamba2(nn.Module):
        def __init__(
            self,
            d_model,
            d_state=128,
            d_conv=4,
            expand=2,
            layer_idx=None,
            device=None,
            dtype=None,
            **kwargs
        ):
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            d_inner = int(expand * d_model)
            self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False, **factory_kwargs)
            self.out_proj = nn.Linear(d_inner, d_model, bias=False, **factory_kwargs)

        def forward(self, hidden_states, inference_params=None, **kwargs):
            xz = self.in_proj(hidden_states)
            x, z = xz.chunk(2, dim=-1)
            return self.out_proj(x * F.silu(z))

    class MHA(nn.Module):
        def __init__(self, d_model, layer_idx=None, device=None, dtype=None, **kwargs):
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            self.proj = nn.Linear(d_model, d_model, **factory_kwargs)

        def forward(self, hidden_states, inference_params=None, **kwargs):
            return self.proj(hidden_states)

    class GatedMLP(nn.Module):
        def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            device=None,
            dtype=None,
            **kwargs
        ):
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            if out_features is None:
                out_features = in_features
            if hidden_features is None:
                hidden_features = 4 * in_features
            self.fc1 = nn.Linear(in_features, 2 * hidden_features, **factory_kwargs)
            self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

        def forward(self, x):
            xz = self.fc1(x)
            x, z = xz.chunk(2, dim=-1)
            return self.fc2(x * F.silu(z))

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
            self.fused_add_norm = fused_add_norm
            self.residual_in_fp32 = residual_in_fp32
            self.norm = norm_cls(dim)
            self.mixer = mixer_cls(dim)
            if mlp_cls is not nn.Identity:
                self.norm2 = norm_cls(dim)
                self.mlp = mlp_cls(dim)
            else:
                self.mlp = None

        def forward(self, hidden_states, residual=None, inference_params=None):
            if self.fused_add_norm:
                is_rms = isinstance(self.norm, RMSNorm)
                fused_fn = rms_norm_fn if is_rms else layer_norm_fn
                hidden_states, residual = fused_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                residual = (
                    (hidden_states + residual)
                    if residual is not None
                    else hidden_states
                )
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
            if self.mlp is not None:
                if self.fused_add_norm:
                    hidden_states, residual = layer_norm_fn(
                        hidden_states,
                        self.norm2.weight,
                        self.norm2.bias,
                        residual=residual,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        eps=self.norm2.eps,
                        is_rms_norm=isinstance(self.norm2, RMSNorm),
                    )
                else:
                    residual = hidden_states + residual
                    hidden_states = self.norm2(
                        residual.to(dtype=self.norm2.weight.dtype)
                    )
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

    class GenerationMixin:
        pass

    # --- build module hierarchy ---

    def _make(name, is_pkg=True):
        m = types.ModuleType(name)
        if is_pkg:
            m.__path__ = []
        sys.modules[name] = m
        return m

    pkg = _make("mamba_ssm")
    models = _make("mamba_ssm.models")
    models_cfg = _make("mamba_ssm.models.config_mamba", is_pkg=False)
    modules = _make("mamba_ssm.modules")
    mod_simple = _make("mamba_ssm.modules.mamba_simple", is_pkg=False)
    mod_mamba2 = _make("mamba_ssm.modules.mamba2", is_pkg=False)
    mod_mha = _make("mamba_ssm.modules.mha", is_pkg=False)
    mod_mlp = _make("mamba_ssm.modules.mlp", is_pkg=False)
    mod_block = _make("mamba_ssm.modules.block", is_pkg=False)
    utils = _make("mamba_ssm.utils")
    utils_gen = _make("mamba_ssm.utils.generation", is_pkg=False)
    ops = _make("mamba_ssm.ops")
    ops_triton = _make("mamba_ssm.ops.triton")
    ops_ln = _make("mamba_ssm.ops.triton.layer_norm", is_pkg=False)

    models_cfg.MambaConfig = MambaConfig
    mod_simple.Mamba = Mamba
    mod_mamba2.Mamba2 = Mamba2
    mod_mha.MHA = MHA
    mod_mlp.GatedMLP = GatedMLP
    mod_block.Block = Block
    utils_gen.GenerationMixin = GenerationMixin
    ops_ln.RMSNorm = RMSNorm
    ops_ln.layer_norm_fn = layer_norm_fn
    ops_ln.rms_norm_fn = rms_norm_fn

    pkg.Mamba = Mamba
    pkg.Mamba2 = Mamba2
