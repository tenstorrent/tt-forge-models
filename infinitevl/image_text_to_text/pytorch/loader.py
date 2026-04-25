# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InfiniteVL model loader implementation for image-text-to-text tasks.
"""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from typing import Optional

# fla/utils.py calls torch.cpu.device(index) as a context manager when Triton is unavailable.
# torch.cpu does not have a `device` attribute in PyTorch 2.9+; provide a no-op.
if not hasattr(torch.cpu, "device"):
    import contextlib

    torch.cpu.device = lambda index: contextlib.nullcontext()


def _compute_default_rope_parameters(config, device=None, seq_len=None, **rope_kwargs):
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

# transformers v5 expects _tied_weights_keys as a dict; older model code uses a list.
# Patch get_expanded_tied_weights_keys to accept both formats.
_orig_get_expanded_tied_weights_keys = PreTrainedModel.get_expanded_tied_weights_keys


def _patched_get_expanded_tied_weights_keys(self, all_submodels=True):
    if isinstance(self._tied_weights_keys, list):
        self._tied_weights_keys = None
    return _orig_get_expanded_tied_weights_keys(self, all_submodels=all_submodels)


PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded_tied_weights_keys

# transformers v5 _init_weights calls module.compute_default_rope_parameters for rope_type=="default"
# but InfiniteVL's custom RotaryEmbedding doesn't have that method. Patch _init_weights to fall back.
_orig_init_weights = PreTrainedModel._init_weights


def _patched_init_weights(self, module):
    if (
        "RotaryEmbedding" in module.__class__.__name__
        and hasattr(module, "original_inv_freq")
        and getattr(module, "rope_type", None) == "default"
        and not hasattr(module, "compute_default_rope_parameters")
    ):
        module.compute_default_rope_parameters = (
            lambda cfg=None: _compute_default_rope_parameters(module.config)
        )
    return _orig_init_weights(self, module)


PreTrainedModel._init_weights = _patched_init_weights

# InfiniteVL uses fla (flash-linear-attention) which dispatches to triton kernels.
# Replace triton-dependent fla ops with pure PyTorch equivalents so the model
# can run on CPU (compile-only) environments without a GPU driver.

from fla.modules.conv.short_conv import ShortConvolution as _ShortConvolution
from fla.modules.fused_norm_gate import FusedRMSNormGated as _FusedRMSNormGated
import fla.ops.gated_delta_rule as _gdr_module
from fla.ops.gated_delta_rule.naive import (
    naive_chunk_gated_delta_rule as _naive_chunk_gdr,
    naive_recurrent_gated_delta_rule as _naive_recurrent_gdr,
)


def _pytorch_short_conv_forward(
    self, x, residual=None, mask=None, cache=None, output_final_state=False, **kwargs
):
    B, T, D = x.shape
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    K = self.kernel_size[0]
    x_t = x.transpose(1, 2)
    x_padded = F.pad(x_t, (K - 1, 0))
    y_t = F.conv1d(x_padded, self.weight, self.bias, groups=D)
    y = y_t.transpose(1, 2)
    if self.activation in ("silu", "swish"):
        y = F.silu(y)
    if residual is not None:
        y = y + residual
    return y, None


_ShortConvolution.forward = _pytorch_short_conv_forward


def _pytorch_fused_rms_norm_gated_forward(
    self, x, g, residual=None, prenorm=False, residual_in_fp32=False
):
    if residual is not None:
        x = x + residual.to(x.dtype)
    residual_out = x if prenorm else None
    orig_dtype = x.dtype
    x_f32 = x.float() if residual_in_fp32 else x.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
    normed = (x_f32 * rms).to(orig_dtype)
    if self.weight is not None:
        normed = normed * self.weight
    if self.activation in ("silu", "swish"):
        gate = F.silu(g)
    else:
        gate = torch.sigmoid(g)
    out = normed * gate
    if prenorm:
        return out, residual_out
    return out


_FusedRMSNormGated.forward = _pytorch_fused_rms_norm_gated_forward


def _patched_chunk_gdr(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    orig_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)
    o, state = _naive_chunk_gdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return o.to(orig_dtype), state


def _patched_fused_recurrent_gdr(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    orig_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)
    o, state = _naive_recurrent_gdr(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return o.to(orig_dtype), state


_gdr_module.chunk_gated_delta_rule = _patched_chunk_gdr
_gdr_module.fused_recurrent_gated_delta_rule = _patched_fused_recurrent_gdr

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
    """Available InfiniteVL model variants for image-text-to-text tasks."""

    INFINITEVL = "infinitevl"


class ModelLoader(ForgeModel):
    """InfiniteVL model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.INFINITEVL: LLMModelConfig(
            pretrained_model_name="hustvl/InfiniteVL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INFINITEVL

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="InfiniteVL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # pad_token_id is absent from InfiniteVLTextConfig; set it before model init
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config.text_config, "pad_token_id"):
            config.text_config.pad_token_id = None

        # The model hardcodes flash_attention_2 in each attention layer's __init__, which
        # requires a CUDA GPU. Intercept the setting to redirect to "sdpa" instead.
        text_config_cls = config.text_config.__class__
        _original_text_setattr = text_config_cls.__setattr__

        def _redirect_attn_impl(self, name, value):
            if name == "_attn_implementation" and value == "flash_attention_2":
                value = "sdpa"
            _original_text_setattr(self, name, value)

        text_config_cls.__setattr__ = _redirect_attn_impl

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
