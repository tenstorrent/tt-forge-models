# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash-heretic-MPOA GGUF model loader implementation for causal language modeling.
"""
import torch
import torch.nn.functional as F
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


def _patch_deepseek_v2_gguf():
    """Patch GGUF_TO_FAST_CONVERTERS and get_gguf_hf_weights_map for deepseek_v2.

    The glm_4_7_flash_gguf loader patches load_gguf_checkpoint to remap
    model_type 'deepseek2' -> 'deepseek_v2', but only registers 'deepseek2'
    in GGUF_TO_FAST_CONVERTERS.  When the tokenizer loads it sees architecture
    'deepseek_v2' and hits KeyError.  Similarly, get_gguf_hf_weights_map needs
    'deepseek2' (not 'deepseek_v2') to find the gguf-py tensor-name map.
    """
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter

    _orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(hf_model, processor=None, model_type=None, num_layers=None, qual_name=""):
        # momix_44 and similar loaders eagerly resolve model_type from config and
        # pass it as an explicit positional arg, so we must handle both cases.
        if model_type == "deepseek_v2":
            model_type = "deepseek2"
        elif model_type is None:
            cfg = getattr(hf_model, "config", None)
            if getattr(cfg, "model_type", None) == "deepseek_v2":
                cfg.model_type = "deepseek2"
                try:
                    return _orig_get_map(hf_model, processor, model_type="deepseek2", num_layers=num_layers, qual_name=qual_name)
                finally:
                    cfg.model_type = "deepseek_v2"
        return _orig_get_map(hf_model, processor, model_type=model_type, num_layers=num_layers, qual_name=qual_name)

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    import transformers.modeling_utils as modeling_utils
    if hasattr(modeling_utils, "get_gguf_hf_weights_map"):
        modeling_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_deepseek_v2_gguf()


def _tt_static_deepseek_v2_moe_forward(experts_module, hidden_states, top_k_index, top_k_weights):
    """Static per-expert masked matmul for DeepseekV2Experts — avoids nonzero() dynamic shapes on XLA.

    Python-int loop is unrolled at trace time into constant-weight F.linear calls.
    """
    dtype = hidden_states.dtype
    out = torch.zeros_like(hidden_states)
    for expert_idx in range(experts_module.num_experts):
        mask = (top_k_index == expert_idx)
        weight = (mask.to(dtype) * top_k_weights.to(dtype)).sum(dim=-1, keepdim=True)
        gate_up = F.linear(hidden_states, experts_module.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_expert = F.linear(experts_module.act_fn(gate) * up, experts_module.down_proj[expert_idx])
        out = out + hidden_expert * weight
    return out.to(dtype)


try:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    ALL_EXPERTS_FUNCTIONS["tt_static_deepseek_v2_moe"] = _tt_static_deepseek_v2_moe_forward

    _orig_get_correct_experts = PreTrainedModel.get_correct_experts_implementation

    def _patched_get_correct_experts(self, requested_experts):
        if requested_experts is not None and requested_experts in ALL_EXPERTS_FUNCTIONS:
            return requested_experts
        return _orig_get_correct_experts(self, requested_experts)

    PreTrainedModel.get_correct_experts_implementation = _patched_get_correct_experts
except Exception:
    pass


def _patch_deepseek_v2_rope():
    """Replace complex-tensor RoPE with real-valued equivalent.

    DeepseekV2YarnRotaryEmbedding.forward uses torch.polar → complex tensors,
    which the TT PJRT backend rejects at buffer_instance.cc with
    "Complex tensor with num_dims == 0 is not supported."
    Replace with equivalent cos/sin computation; patch apply_rotary_emb to
    use real arithmetic instead of view_as_complex / view_as_real.
    """
    import transformers.models.deepseek_v2.modeling_deepseek_v2 as _dsv2

    @torch.no_grad()
    def _patched_yarn_rope_forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
        cos = torch.cos(freqs) * self.attention_scaling
        sin = torch.sin(freqs) * self.attention_scaling
        return (cos, sin)

    def _patched_apply_rotary_emb(xq, xk, freqs_cis):
        if isinstance(freqs_cis, tuple):
            cos, sin = freqs_cis
        else:
            # Fallback: extract cos/sin from a complex tensor if already created
            cos = freqs_cis.real
            sin = freqs_cis.imag
        # cos/sin: [batch, seq_len, head_dim/2] → broadcast over heads
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        # xq/xk: [batch, heads, seq_len, qk_rope_head_dim]
        xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
        xq_out = torch.stack(
            [xq_r[..., 0] * cos - xq_r[..., 1] * sin,
             xq_r[..., 0] * sin + xq_r[..., 1] * cos],
            dim=-1,
        ).flatten(-2).type_as(xq)
        xk_out = torch.stack(
            [xk_r[..., 0] * cos - xk_r[..., 1] * sin,
             xk_r[..., 0] * sin + xk_r[..., 1] * cos],
            dim=-1,
        ).flatten(-2).type_as(xk)
        return xq_out, xk_out

    _dsv2.DeepseekV2RotaryEmbedding.forward = _patched_yarn_rope_forward
    _dsv2.apply_rotary_emb = _patched_apply_rotary_emb


_patch_deepseek_v2_rope()


class ModelVariant(StrEnum):
    """Available GLM-4.7-Flash-heretic-MPOA GGUF model variants for causal language modeling."""

    GLM_4_7_FLASH_HERETIC_MPOA_GGUF = "4.7_Flash_heretic_MPOA_GGUF"


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash-heretic-MPOA GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_HERETIC_MPOA_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GLM-4.7-Flash-heretic-MPOA-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_HERETIC_MPOA_GGUF

    GGUF_FILE = "GLM-4.7-Flash-heretic-MPOA.i1-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="GLM-4.7-Flash-heretic-MPOA GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["ignore_mismatched_sizes"] = True

        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        # MLA uses num_attention_heads for both q and k/v heads internally;
        # GGUF stores head_count_kv=1 (latent rank marker) which would cause
        # GQA expansion to multiply heads by num_attention_heads before SDPA.
        if (
            hasattr(config, "num_key_value_heads")
            and hasattr(config, "num_attention_heads")
            and config.num_key_value_heads < config.num_attention_heads
        ):
            config.num_key_value_heads = config.num_attention_heads
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        # Use static per-expert masked matmul: avoids nonzero() dynamic shapes
        # (original forward) and histc/gather issues (grouped_mm / batched_mm).
        config._experts_implementation = "tt_static_deepseek_v2_moe"
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

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "up_proj"):
                # Dense MLP
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")
            # MoE layers use 3-D Parameter tensors; skip for single-device.

            attn = layer.self_attn
            if hasattr(attn, "q_proj"):
                # Standard GQA attention
                shard_specs[attn.q_proj.weight] = ("model", "batch")
                shard_specs[attn.k_proj.weight] = ("model", "batch")
                shard_specs[attn.v_proj.weight] = ("model", "batch")
            elif hasattr(attn, "q_b_proj"):
                # MLA attention (DeepseekV2): no q/k/v projections; use compressed heads
                shard_specs[attn.q_b_proj.weight] = ("model", "batch")
                if hasattr(attn, "kv_b_proj"):
                    shard_specs[attn.kv_b_proj.weight] = ("model", "batch")
            if hasattr(attn, "o_proj"):
                shard_specs[attn.o_proj.weight] = ("batch", "model")

        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
