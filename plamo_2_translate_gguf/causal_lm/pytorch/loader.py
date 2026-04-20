# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PLaMo 2 Translate GGUF model loader implementation for causal language modeling.

PLaMo 2 uses a hybrid Mamba2/Attention architecture whose GGUF format is not
yet supported by transformers' built-in GGUF loader.  We fall back to the
non-GGUF source repo (pfnet/plamo-2-translate) which ships custom modeling
code via trust_remote_code.
"""
import importlib
import sys
import types

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

BASE_MODEL_REPO = "pfnet/plamo-2-translate"


def _install_causal_conv1d_shim():
    """Install a pure-PyTorch shim for causal_conv1d when the CUDA package is unavailable."""
    if "causal_conv1d" in sys.modules:
        return

    def causal_conv1d_update_ref(x, conv_state, weight, activation="silu"):
        x_sq = x.squeeze(-1) if x.dim() == 3 else x
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x_sq
        window = torch.cat([conv_state, x_sq.unsqueeze(-1)], dim=-1)
        window = window[:, :, -weight.shape[-1] :]
        out = torch.sum(window * weight.unsqueeze(0), dim=-1)
        if activation == "silu":
            out = F.silu(out)
        return out.unsqueeze(-1) if x.dim() == 3 else out

    def causal_conv1d_ref(
        x,
        weight,
        initial_states=None,
        return_final_states=False,
        activation="silu",
        seq_idx=None,
    ):
        batch, dim, seq_len = x.shape
        width = weight.shape[-1]
        if initial_states is not None:
            conv_state = initial_states.clone()
        else:
            conv_state = x.new_zeros(batch, dim, width - 1)

        x_padded = torch.cat([conv_state, x], dim=-1)
        weight_flip = weight.unsqueeze(0).flip(-1)
        out = F.conv1d(x_padded, weight_flip.reshape(dim, 1, width), groups=dim)[
            :, :, :seq_len
        ]
        if activation == "silu":
            out = F.silu(out)
        final_state = (
            x_padded[:, :, -(width - 1) :].clone() if return_final_states else None
        )
        if return_final_states:
            return out, final_state
        return out

    def causal_conv1d_fn(
        x,
        weight,
        initial_states=None,
        return_final_states=False,
        activation="silu",
        seq_idx=None,
    ):
        return causal_conv1d_ref(
            x,
            weight,
            initial_states=initial_states,
            return_final_states=return_final_states,
            activation=activation,
            seq_idx=seq_idx,
        )

    iface = types.ModuleType("causal_conv1d.causal_conv1d_interface")
    iface.causal_conv1d_update_ref = causal_conv1d_update_ref
    iface.causal_conv1d_ref = causal_conv1d_ref
    iface.causal_conv1d_fn = causal_conv1d_fn
    iface.causal_conv1d_update = causal_conv1d_update_ref

    pkg = types.ModuleType("causal_conv1d")
    pkg.__path__ = []
    pkg.causal_conv1d_interface = iface

    sys.modules["causal_conv1d"] = pkg
    sys.modules["causal_conv1d.causal_conv1d_interface"] = iface


def _install_mamba_ssm_selective_state_update_shim():
    """Add selective_state_update_ref to the mamba_ssm shim if missing."""
    mod_name = "mamba_ssm.ops.triton.selective_state_update"
    mod = sys.modules.get(mod_name)
    if mod is None:
        return
    if hasattr(mod, "selective_state_update_ref"):
        return

    def selective_state_update_ref(
        state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
    ):
        out_dtype = x.dtype
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = F.softplus(dt)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)
        state.copy_(state * dA + x.unsqueeze(-1) * dB)
        y = torch.sum(state * C.unsqueeze(-2), dim=-1)
        if D is not None:
            y = y + D * x
        if z is not None:
            y = y * F.silu(z)
        return y.to(out_dtype)

    mod.selective_state_update_ref = selective_state_update_ref
    if not hasattr(mod, "selective_state_update"):
        mod.selective_state_update = selective_state_update_ref


def _install_all_shims():
    _install_causal_conv1d_shim()
    _install_mamba_ssm_selective_state_update_shim()


class ModelVariant(StrEnum):
    """Available PLaMo 2 Translate GGUF model variants for causal language modeling."""

    PLAMO_2_TRANSLATE_GGUF = "TRANSLATE_GGUF"


class ModelLoader(ForgeModel):
    """PLaMo 2 Translate GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PLAMO_2_TRANSLATE_GGUF: LLMModelConfig(
            pretrained_model_name=BASE_MODEL_REPO,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PLAMO_2_TRANSLATE_GGUF

    sample_text = (
        "Translate the following English text to Japanese: The weather is nice today."
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PLaMo 2 Translate GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _install_all_shims()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        _install_all_shims()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
