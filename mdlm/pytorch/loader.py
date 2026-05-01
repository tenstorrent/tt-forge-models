# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MDLM (Masked Diffusion Language Model) loader implementation.
"""
import sys
import types
import torch
import torch.nn.functional as F
from typing import Optional

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def _apply_rotary_emb_qkv_(qkv, cos, sin):
    """PyTorch replacement for flash_attn.layers.rotary.apply_rotary_emb_qkv_.

    qkv: [b, s, 3, h, d], cos/sin: [s, d//2]
    Applies RoPE in-place to q (index 0) and k (index 1); v is unchanged.
    """
    d2 = cos.shape[-1]
    # cos/sin may be on CPU (cached from a prior run) while qkv is on XLA;
    # move them to match qkv's device before any arithmetic.
    cos = cos.to(device=qkv.device).unsqueeze(-2)  # [s, 1, d//2]
    sin = sin.to(device=qkv.device).unsqueeze(-2)  # [s, 1, d//2]
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    q1, q2 = q[..., :d2], q[..., d2:]
    k1, k2 = k[..., :d2], k[..., d2:]
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    v = qkv[:, :, 2]
    return torch.stack([q_rot, k_rot, v], dim=2)


def _flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p, causal=False):
    """PyTorch replacement for flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func.

    qkv: [total, 3, h, d], cu_seqlens: int32 cumulative lengths, max_seqlen: int
    Assumes uniform sequence lengths (cu_seqlens = [0, L, 2L, ...]).
    """
    total, _, h, d = qkv.shape
    batch_size = cu_seqlens.shape[0] - 1
    qkv_b = qkv.view(batch_size, max_seqlen, 3, h, d)
    q = qkv_b[:, :, 0].transpose(1, 2)  # [b, h, s, d]
    k = qkv_b[:, :, 1].transpose(1, 2)
    v = qkv_b[:, :, 2].transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2).reshape(total, h, d)


def _inject_flash_attn_stub():
    """Inject a minimal flash_attn stub so transformers check_imports succeeds
    and the model's unconditional `import flash_attn` resolves to our PyTorch
    implementations instead of raising ImportError."""
    if "flash_attn" in sys.modules:
        return

    from importlib.machinery import ModuleSpec

    def _make_mod(name):
        mod = types.ModuleType(name)
        mod.__spec__ = ModuleSpec(name, None, origin="stub")
        mod.__package__ = name.rpartition(".")[0] or name
        return mod

    fa = _make_mod("flash_attn")
    fa_layers = _make_mod("flash_attn.layers")
    fa_rotary = _make_mod("flash_attn.layers.rotary")
    fa_iface = _make_mod("flash_attn.flash_attn_interface")

    fa_rotary.apply_rotary_emb_qkv_ = _apply_rotary_emb_qkv_
    fa_iface.flash_attn_varlen_qkvpacked_func = _flash_attn_varlen_qkvpacked_func

    fa.layers = fa_layers
    fa_layers.rotary = fa_rotary
    fa.flash_attn_interface = fa_iface

    sys.modules["flash_attn"] = fa
    sys.modules["flash_attn.layers"] = fa_layers
    sys.modules["flash_attn.layers.rotary"] = fa_rotary
    sys.modules["flash_attn.flash_attn_interface"] = fa_iface
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available MDLM model variants."""

    MDLM_OWT = "OWT"


class ModelLoader(ForgeModel):
    """MDLM model loader implementation."""

    _VARIANTS = {
        ModelVariant.MDLM_OWT: LLMModelConfig(
            pretrained_model_name="kuleshov-group/mdlm-owt",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MDLM_OWT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MDLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        _inject_flash_attn_stub()

        # Transformers 5.x requires PreTrainedModel.__init__ subclasses to call
        # self.post_init() to initialize all_tied_weights_keys and related attrs.
        # The remote MDLM class predates this requirement, so patch it here.
        mdlm_cls = get_class_from_dynamic_module(
            "modeling_mdlm.MDLM", pretrained_model_name
        )
        _orig_mdlm_init = mdlm_cls.__init__
        def _patched_mdlm_init(self_m, config):
            _orig_mdlm_init(self_m, config)
            if not hasattr(self_m, "all_tied_weights_keys"):
                self_m.post_init()
        mdlm_cls.__init__ = _patched_mdlm_init

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        # modeling_mdlm.py defines `modulate` twice (lines ~53 and ~133).
        # The second definition (x*(1+scale.unsqueeze(1))+shift.unsqueeze(1))
        # shadows the first.  modulate_fused is @torch.jit.script and captured
        # the first version at compile time, but TorchDynamo re-executes its
        # Python source using the CURRENT global `modulate` (second version),
        # which inserts an extra dimension and causes the DDiTBlock rearrange to
        # receive a 4D tensor.  Replace modulate_fused with a plain Python
        # function so Dynamo sees the correct (no-unsqueeze) logic directly.
        _remote_mod = sys.modules.get(type(model.backbone).__module__)
        if _remote_mod is not None:
            def _modulate_fused_fixed(x, shift, scale):
                return x * (1 + scale) + shift
            _remote_mod.modulate_fused = _modulate_fused_fixed

        # The remote TimestepEmbedder.timestep_embedding() explicitly creates
        # float32 tensors, causing a dtype mismatch when model weights are
        # bfloat16.  Wrap sigma_map.forward to cast t_freq to the layer dtype.
        _sigma_map = model.backbone.sigma_map
        def _sigma_fwd(self, t):
            t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
            weight_dtype = next(self.parameters()).dtype
            return self.mlp(t_freq.to(weight_dtype))
        _sigma_map.__class__.forward = _sigma_fwd

        # DITBackbone.forward uses torch.cuda.amp.autocast(dtype=bfloat16), which
        # raises on machines whose CUDA device doesn't support bfloat16 (and is a
        # no-op on TT/CPU anyway since the model is already in bfloat16).  Replace
        # with torch.amp.autocast so the CPU comparison run succeeds.
        _dit_cls = type(model.backbone)
        def _dit_fwd(self, indices, sigma, output_hidden_states=False):
            if not self.config.time_conditioning:
                sigma = torch.zeros_like(sigma)
            all_hidden_states = []
            x = self.vocab_embed(indices)
            if output_hidden_states:
                all_hidden_states.append(x)
            c = F.silu(self.sigma_map(sigma))
            rotary_cos_sin = self.rotary_emb(x)
            device_type = "cpu" if not x.is_cuda else "cuda"
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
                    if output_hidden_states:
                        all_hidden_states.append(x)
                logits = self.output_layer(x, c)
            return logits, all_hidden_states
        _dit_cls.forward = _dit_fwd

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        timesteps = torch.zeros(1)

        return {
            "input_ids": inputs["input_ids"],
            "timesteps": timesteps,
        }

    def decode_output(self, outputs):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_ids = logits.argmax(dim=-1)
        decoded = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        print("Decoded output:", decoded)
        return decoded
