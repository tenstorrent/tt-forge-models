# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.
"""
import sys
import types
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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


def _create_mamba_ssm_stub():
    """Inject a minimal mamba_ssm stub with a pure-PyTorch rmsnorm_fn fallback.

    NemotronH unconditionally imports rmsnorm_fn from mamba_ssm at module level.
    mamba_ssm is a CUDA-only Triton library that cannot be installed on CPU-only
    systems.  On CPU the model's forward path automatically uses torch_forward
    (no Triton kernels), so we only need the import to succeed.
    """
    if "mamba_ssm" in sys.modules:
        return

    def rmsnorm_fn(
        x,
        weight,
        bias=None,
        residual=None,
        z=None,
        prenorm=False,
        residual_in_fp32=False,
        eps=1e-6,
        dropout_p=0.0,
        group_size=None,
        norm_before_gate=True,
    ):
        """Pure-PyTorch grouped RMSNorm with optional SiLU gate."""
        if residual is not None:
            x = x + residual.to(x.dtype)
        orig_dtype = x.dtype
        x_f = x.float()
        if group_size is not None and group_size > 0:
            orig_shape = x_f.shape
            hidden = orig_shape[-1]
            n_grp = hidden // group_size
            x_g = x_f.reshape(*orig_shape[:-1], n_grp, group_size)
            var = x_g.pow(2).mean(-1, keepdim=True)
            x_normed = (x_g * torch.rsqrt(var + eps)).reshape(orig_shape)
        else:
            var = x_f.pow(2).mean(-1, keepdim=True)
            x_normed = x_f * torch.rsqrt(var + eps)
        if weight is not None:
            x_normed = x_normed * weight.float()
        if bias is not None:
            x_normed = x_normed + bias.float()
        if z is not None:
            gate = (
                F.silu(z.float()) if not norm_before_gate else torch.sigmoid(z.float())
            )
            x_normed = x_normed * gate
        out = x_normed.to(orig_dtype)
        if prenorm:
            return out, x.to(orig_dtype)
        return out

    mamba_ssm = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    triton_mod = types.ModuleType("mamba_ssm.ops.triton")
    layernorm_gated = types.ModuleType("mamba_ssm.ops.triton.layernorm_gated")
    layernorm_gated.rmsnorm_fn = rmsnorm_fn
    ssm_state_mod = types.ModuleType("mamba_ssm.ops.triton.selective_state_update")
    ssm_state_mod.selective_state_update = None
    ssd_combined = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
    ssd_combined.mamba_chunk_scan_combined = None
    ssd_combined.mamba_split_conv1d_scan_combined = None

    mamba_ssm.ops = ops
    ops.triton = triton_mod
    triton_mod.layernorm_gated = layernorm_gated
    triton_mod.selective_state_update = ssm_state_mod
    triton_mod.ssd_combined = ssd_combined

    sys.modules["mamba_ssm"] = mamba_ssm
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.triton"] = triton_mod
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"] = layernorm_gated
    sys.modules["mamba_ssm.ops.triton.selective_state_update"] = ssm_state_mod
    sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd_combined


def _patch_transformers_nemotron_h_moe_gguf():
    """Monkey-patch transformers to add nemotron_h_moe GGUF architecture support.

    Transformers 5.x lacks GGUF loading support for the 'nemotron_h_moe'
    architecture used by NVIDIA Nemotron-H MoE models.  The native HF
    model_type is 'nemotron_h' (NemotronHForCausalLM via trust_remote_code),
    so after reading the GGUF config we remap model_type accordingly and fill
    in values that are absent from the GGUF metadata.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        NemotronTensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h_moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "rope.dimension_count": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "norm_eps",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "expert_used_count": "num_experts_per_tok",
        "expert_count": "n_routed_experts",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
        "expert_shared_count": "n_shared_experts",
        "moe_latent_size": "moe_latent_size",
        "ssm.conv_kernel": "conv_kernel",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "n_groups",
        "ssm.inner_size": None,
        "ssm.time_step_rank": None,
    }

    TENSOR_PROCESSORS["nemotron_h"] = NemotronTensorProcessor

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "nemotron_h_moe":
            cfg = result["config"]
            cfg["model_type"] = "nemotron_h"
            # GGUF stores 0 for num_key_value_heads on hybrid models; use known value
            if cfg.get("num_key_value_heads", 0) == 0:
                cfg["num_key_value_heads"] = 2
            # ssm.inner_size = mamba_num_heads * mamba_head_dim (8192 = 128 * 64)
            cfg.setdefault("mamba_num_heads", 128)
            cfg.setdefault("mamba_head_dim", 64)
            # Hybrid layer pattern from the official model config
            cfg.setdefault(
                "hybrid_override_pattern",
                "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
                "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
            )
            cfg.setdefault("mtp_hybrid_override_pattern", "*E")
            cfg.setdefault("chunk_size", 128)
            cfg.setdefault("expand", 2)
            cfg.setdefault("mlp_hidden_act", "relu2")
            cfg.setdefault("mamba_hidden_act", "silu")
            cfg.setdefault("norm_topk_prob", True)
            cfg.setdefault("residual_in_fp32", False)
            cfg.setdefault("rescale_prenorm_residual", True)
            cfg.setdefault("mlp_bias", False)
            cfg.setdefault("mamba_proj_bias", False)
            cfg.setdefault("use_bias", False)
            cfg.setdefault("use_conv_bias", True)
            cfg.setdefault("use_mamba_kernels", True)
            cfg.setdefault("mamba_ssm_cache_dtype", "float32")
            cfg.setdefault("num_logits_to_keep", 1)
            cfg.setdefault("num_nextn_predict_layers", 1)
            cfg.setdefault("partial_rotary_factor", 1.0)
            cfg.setdefault("time_step_floor", 0.0001)
            cfg.setdefault("time_step_max", 0.1)
            cfg.setdefault("time_step_min", 0.001)
            cfg.setdefault("moe_shared_expert_overlap", False)
            cfg.setdefault("hidden_dropout", 0.0)
            cfg.setdefault("attention_dropout", 0.0)
            cfg.setdefault("attention_bias", False)
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    try:
        import transformers.tokenization_utils_tokenizers as tok_tokenizers

        if hasattr(tok_tokenizers, "load_gguf_checkpoint"):
            tok_tokenizers.load_gguf_checkpoint = patched_load_gguf_checkpoint
    except Exception:
        pass

    # Use NEMOTRON_H_MOE arch for tensor name mapping so that MoE and SSM
    # tensor names are included when building the GGUF→HF weight map.
    orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, **kw):
        mt = hf_model.config.model_type if model_type is None else model_type
        if mt == "nemotron_h":
            model_type = "nemotron_h_moe"
        return orig_get_weights_map(hf_model, processor, model_type=model_type, **kw)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_create_mamba_ssm_stub()
_patch_transformers_nemotron_h_moe_gguf()


class ModelVariant(StrEnum):
    """Available AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF = "AesSedai_3_Super_120B_A12B_GGUF"


class ModelLoader(ForgeModel):
    """AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF

    GGUF_FILE = (
        "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf"
    )

    sample_text = "Give me a short introduction to large language models."

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
            model="AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF",
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
        tokenizer_kwargs["trust_remote_code"] = True

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
        model_kwargs["trust_remote_code"] = True

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                gguf_file=self.GGUF_FILE,
                trust_remote_code=True,
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            trust_remote_code=True,
        )
        return self.config
