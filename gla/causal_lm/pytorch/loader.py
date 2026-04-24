# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLA (Gated Linear Attention) model loader implementation for causal language modeling.
"""
import torch
import torch.nn.functional as F
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

try:
    import contextlib
    import fla.utils as _fla_utils
    import fla.layers.gla as _fla_layers_gla
    import fla.modules.layernorm as _fla_layernorm
    import fla.modules.fused_norm_gate as _fla_norm_gate
    from fla.models.gla.configuration_gla import GLAConfig
    from fla.models.gla.modeling_gla import GLAForCausalLM
    from fla.ops.gla.naive import naive_recurrent_gla as _naive_recurrent_gla

    # transformers 5.x changed _tied_weights_keys from list to dict format
    GLAForCausalLM._tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}
    AutoConfig.register("gla", GLAConfig, exist_ok=True)
    AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM, exist_ok=True)

    # fla 0.5.0 bug: custom_device_ctx calls torch.cpu.device() which doesn't exist;
    # on CPU (index=None) return a no-op context manager instead.
    def _cpu_safe_device_ctx(index):
        if index is None or not hasattr(_fla_utils.device_torch_lib, "device"):
            return contextlib.nullcontext()
        return _fla_utils.device_torch_lib.device(index)

    _fla_utils.custom_device_ctx = _cpu_safe_device_ctx

    # fla norm ops use triton kernels which require a GPU driver; replace with
    # pure PyTorch implementations for CPU execution.
    def _torch_layer_norm_fwd(
        x,
        weight,
        bias,
        eps=1e-5,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        is_rms_norm=False,
        num_groups=1,
    ):
        if residual is not None:
            x = x + residual
        res_out = (
            x.to(residual_dtype)
            if (
                residual is not None
                or (residual_dtype is not None and residual_dtype != x.dtype)
            )
            else None
        )
        x_f = x.float()
        if is_rms_norm:
            rstd = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
            y = x_f * rstd
            mean = None
        else:
            mean_val = x_f.mean(-1, keepdim=True)
            rstd = torch.rsqrt((x_f - mean_val).pow(2).mean(-1, keepdim=True) + eps)
            y = (x_f - mean_val) * rstd
            mean = mean_val.squeeze(-1)
        if weight is not None:
            y = y * weight.float()
        if bias is not None:
            y = y + bias.float()
        return y.to(out_dtype or x.dtype), mean, rstd.squeeze(-1), res_out

    def _torch_layer_norm_gated_fwd(
        x,
        g,
        weight,
        bias,
        activation="swish",
        eps=1e-5,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        is_rms_norm=False,
    ):
        if residual is not None:
            x = x + residual
        residual_out = (
            x.to(residual_dtype)
            if (
                residual is not None
                or (residual_dtype is not None and residual_dtype != x.dtype)
            )
            else None
        )
        x_f = x.float()
        if is_rms_norm:
            rstd = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
            y = x_f * rstd
            mean = None
        else:
            mean_val = x_f.mean(-1, keepdim=True)
            rstd = torch.rsqrt((x_f - mean_val).pow(2).mean(-1, keepdim=True) + eps)
            y = (x_f - mean_val) * rstd
            mean = mean_val.squeeze(-1)
        if weight is not None:
            y = y * weight.float()
        if bias is not None:
            y = y + bias.float()
        if activation in ("swish", "silu"):
            gate = F.silu(g.float())
        elif activation == "sigmoid":
            gate = torch.sigmoid(g.float())
        else:
            gate = g.float()
        y = (y * gate).to(out_dtype or x.dtype)
        return y, mean, rstd.squeeze(-1), residual_out

    _fla_layernorm.layer_norm_fwd = _torch_layer_norm_fwd
    _fla_norm_gate.layer_norm_gated_fwd = _torch_layer_norm_gated_fwd

    # GLA attention ops use triton; replace with the naive pure-PyTorch reference.
    def _naive_gla_op(
        q,
        k,
        v,
        gk=None,
        gv=None,
        scale=None,
        initial_state=None,
        output_final_state=False,
        reverse=False,
        cu_seqlens=None,
    ):
        return _naive_recurrent_gla(
            q,
            k,
            v,
            gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    _fla_layers_gla.fused_recurrent_gla = _naive_gla_op
    _fla_layers_gla.fused_chunk_gla = _naive_gla_op
    _fla_layers_gla.chunk_gla = _naive_gla_op

except ImportError:
    pass

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


class ModelVariant(StrEnum):
    """Available GLA model variants for causal language modeling."""

    GLA_340M_15B = "340M-15B"


class ModelLoader(ForgeModel):
    """GLA model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLA_340M_15B: LLMModelConfig(
            pretrained_model_name="fla-hub/gla-340M-15B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLA_340M_15B

    sample_text = "Hello, I am"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GLA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
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
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        tokenized_inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )

        inputs = {
            "input_ids": tokenized_inputs.input_ids,
            "attention_mask": tokenized_inputs.attention_mask,
        }

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        token_ids = torch.argmax(logits, dim=-1)
        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        return decoded[0] if len(decoded) == 1 else decoded
