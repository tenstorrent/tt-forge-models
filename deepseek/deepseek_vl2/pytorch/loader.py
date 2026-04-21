# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek VL2 model loader implementation for multimodal vision-language tasks.
"""

import sys
import types
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import Optional


class _LlamaAttentionCompat(torch.nn.Module):
    """Compatibility LlamaAttention matching the transformers 4.x forward API used by deepseek-vl2 remote code."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        bias = getattr(config, "attention_bias", False)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.layer_idx = layer_idx

        self.q_proj = torch.nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = torch.nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = torch.nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=bias
        )

        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotary_emb(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        B, L, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = L + (
            past_key_value[0].shape[-2] if past_key_value is not None else 0
        )
        cos, sin = self._rotary_emb(
            kv_seq_len, hidden_states.device, hidden_states.dtype
        )

        if position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        else:
            cos = cos[:L].unsqueeze(0).unsqueeze(0)
            sin = sin[:L].unsqueeze(0).unsqueeze(0)

        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v) if use_cache else None

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.head_dim**-0.5,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, present_key_value


def _patch_transformers_for_deepseek_vl2():
    """Patch transformers 5.x API for deepseek-vl2 remote code compatibility."""
    # Inject xformers stub so siglip_vit.py loads without the real package.
    if "xformers" not in sys.modules:

        def memory_efficient_attention(
            q, k, v, attn_bias=None, p=0.0, scale=None, **kwargs
        ):
            # xformers [B, N, H, D] -> PyTorch SDPA [B, H, N, D]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=p if p > 0 else 0.0)
            return out.transpose(1, 2)

        xformers_mod = types.ModuleType("xformers")
        xformers_ops = types.ModuleType("xformers.ops")
        xformers_ops.memory_efficient_attention = memory_efficient_attention
        xformers_mod.ops = xformers_ops
        sys.modules["xformers"] = xformers_mod
        sys.modules["xformers.ops"] = xformers_ops

    # is_torch_fx_available was removed in transformers 5.x.
    import transformers.utils.import_utils as _import_utils

    if not hasattr(_import_utils, "is_torch_fx_available"):
        _import_utils.is_torch_fx_available = lambda: False

    # LlamaAttention in transformers 5.x has a different forward API; patch in the compat class.
    import transformers.models.llama.modeling_llama as _llama_mod

    _llama_mod.LlamaAttention = _LlamaAttentionCompat
    if not hasattr(_llama_mod, "LlamaFlashAttention2"):
        _llama_mod.LlamaFlashAttention2 = _LlamaAttentionCompat


_patch_transformers_for_deepseek_vl2()

from ....tools.utils import get_file
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
    """Available DeepSeek VL2 model variants."""

    DEEPSEEK_VL2_TINY = "Tiny"


class ModelLoader(ForgeModel):
    """DeepSeek VL2 model loader implementation for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_VL2_TINY: ModelConfig(
            pretrained_model_name="Isotr0py/deepseek-vl2-tiny",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_VL2_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DeepSeek VL2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
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

        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        conversation = [
            {
                "role": "user",
                "content": "<image>\nDescribe this image.",
            },
            {"role": "assistant", "content": ""},
        ]

        batch = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        )

        images = batch.images
        if dtype_override is not None and images is not None:
            images = images.to(dtype_override)

        return {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "images": images,
            "images_seq_mask": batch.images_seq_mask,
            "images_spatial_crop": batch.images_spatial_crop,
            "use_cache": False,
        }
