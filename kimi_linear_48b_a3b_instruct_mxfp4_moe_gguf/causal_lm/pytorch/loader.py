# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi Linear 48B A3B Instruct MXFP4 MOE GGUF model loader implementation for causal language modeling.
"""
import re
import importlib.metadata

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.utils.import_utils as _import_utils

_orig_is_gguf_available = _gguf_utils.is_gguf_available


def _patched_is_gguf_available(min_version=_import_utils.GGUF_MIN_VERSION):
    """Refresh package mapping so gguf installed mid-process is recognised."""
    if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        _import_utils.PACKAGE_DISTRIBUTION_MAPPING.update(
            importlib.metadata.packages_distributions()
        )
    return _orig_is_gguf_available(min_version)


_gguf_utils.is_gguf_available = _patched_is_gguf_available

SOURCE_REPO = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


def _patch_transformers_compat():
    """Add missing symbols to transformers.utils.generic for Kimi modeling code."""
    import functools
    import transformers.utils.generic as _generic

    if not hasattr(_generic, "OutputRecorder"):
        from transformers.modeling_utils import OutputRecorder

        _generic.OutputRecorder = OutputRecorder

    if not hasattr(_generic, "check_model_inputs"):

        def check_model_inputs(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        _generic.check_model_inputs = check_model_inputs


_patch_transformers_compat()


def _patch_fla_cpu_fallbacks():
    """Replace triton-dependent fla operations with pure PyTorch for CPU-only environments."""
    import contextlib
    import torch.nn.functional as F

    try:
        import fla.utils as fla_utils
    except ImportError:
        return

    if hasattr(fla_utils.device_torch_lib, "device"):
        return

    fla_utils.custom_device_ctx = lambda index: contextlib.nullcontext()

    def _causal_conv1d_torch(
        x,
        weight,
        bias=None,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation=None,
        **kwargs,
    ):
        B, T, D = x.shape
        W = weight.shape[-1]
        x_t = x.transpose(1, 2)
        x_padded = F.pad(x_t, (W - 1, 0))
        y = F.conv1d(x_padded, weight.unsqueeze(1), bias=bias, groups=D)
        if activation in ("silu", "swish"):
            y = F.silu(y)
        y = y.transpose(1, 2)
        final_state = None
        if output_final_state:
            final_state = F.pad(x_t, (max(0, W - 1 - T), 0))[:, :, -(W - 1) :]
        return y, final_state

    from fla.modules.conv.short_conv import ShortConvolution
    from einops import rearrange as _rearrange

    def _shortconv_forward(self, x, cache=None, output_final_state=False, **kwargs):
        return _causal_conv1d_torch(
            x=x,
            weight=_rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
        )

    ShortConvolution.forward = _shortconv_forward

    def _fused_kda_gate_torch(g, A_log, head_dim=None, g_bias=None, **kwargs):
        if isinstance(head_dim, int):
            g = g.unflatten(-1, (-1, head_dim))
        if g_bias is not None:
            g = g + g_bias.view(g.shape[-2:])
        return (-A_log.exp() * F.softplus(g)).float()

    import fla.ops.kda.gate as _gate_mod
    import fla.ops.kda as _kda_mod

    _gate_mod.fused_kda_gate = _fused_kda_gate_torch

    def _chunk_kda_torch(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        B, T, H, K = q.shape
        V = v.shape[-1]
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q.float(), p=2, dim=-1)
            k = F.normalize(k.float(), p=2, dim=-1)
        if scale is None:
            scale = K**-0.5
        scores = torch.einsum("bthk,bshk->bths", q.float() * scale, k.float())
        scores = scores.softmax(dim=-1)
        o = torch.einsum("bths,bshv->bthv", scores, v.float()).to(v.dtype)
        final_state = None
        if output_final_state:
            final_state = torch.zeros(B, H, K, V, dtype=v.dtype, device=v.device)
        return o, final_state

    import fla.ops.kda.chunk as _chunk_mod
    import fla.ops.kda.fused_recurrent as _recurrent_mod

    _chunk_mod.chunk_kda = _chunk_kda_torch
    _kda_mod.chunk_kda = _chunk_kda_torch
    _recurrent_mod.fused_recurrent_kda = _chunk_kda_torch
    _kda_mod.fused_recurrent_kda = _chunk_kda_torch

    from fla.modules.fused_norm_gate import FusedRMSNormGated

    def _fused_rmsnorm_gated_forward(
        self, x, g, residual=None, prenorm=False, **kwargs
    ):
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_flat = x.reshape(-1, x.shape[-1]).float()
        g_flat = g.reshape(-1, g.shape[-1]).float()
        var = x_flat.pow(2).mean(-1, keepdim=True)
        x_normed = x_flat * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            x_normed = x_normed * self.weight.float()
        if self.activation in ("swish", "silu"):
            gate = F.silu(g_flat)
        elif self.activation == "sigmoid":
            gate = torch.sigmoid(g_flat)
        else:
            gate = g_flat
        y = (x_normed * gate).to(orig_dtype).reshape(orig_shape)
        return y

    FusedRMSNormGated.forward = _fused_rmsnorm_gated_forward


_patch_fla_cpu_fallbacks()


def _patch_transformers_kimi_linear_gguf():
    """Monkey-patch transformers to add kimi-linear GGUF architecture support."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        TensorProcessor,
        GGUFTensor,
    )

    if "kimi-linear" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("kimi-linear")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["kimi-linear"] = {
        "context_length": "model_max_length",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "qk_rope_head_dim",
        "rope.freq_base": "rope_theta",
        "attention.key_length": "_gguf_attn_key_length",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "_gguf_head_count_kv",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.kv_lora_rank": "kv_lora_rank",
        "attention.key_length_mla": "_gguf_key_length_mla",
        "attention.value_length": "head_dim",
        "attention.value_length_mla": "v_head_dim",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_token",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_count": "num_shared_experts",
        "leading_dense_block_count": "first_k_dense_replace",
        "expert_weights_scale": "routed_scaling_factor",
        "ssm.conv_kernel": "_gguf_ssm_conv_kernel",
        "kda.head_dim": "_gguf_kda_head_dim",
        "expert_group_used_count": "topk_group",
        "expert_gating_func": "_gguf_gating_func",
    }

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "gpt2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["kimi-linear"] = GGUF_TO_FAST_CONVERTERS["gpt2"]
        GGUF_TO_FAST_CONVERTERS["kimi_linear"] = GGUF_TO_FAST_CONVERTERS["gpt2"]

    class KimiLinearTensorProcessor(TensorProcessor):
        HF_EXPERT_PATTERN = re.compile(r"block_sparse_moe\.experts\.\d+\.")
        GGUF_MOE_EXPS_PATTERN = re.compile(
            r"(?P<name>.*\.ffn_(?P<w>gate|down|up)_exps)\.weight$"
        )
        DT_BIAS_PATTERN = re.compile(r"model\.layers\.(?P<bid>\d+)\.self_attn\.dt_bias")
        E_SCORE_BIAS_PATTERN = re.compile(
            r"model\.layers\.(?P<bid>\d+)\.block_sparse_moe\.gate\."
            r"e_score_correction_bias"
        )

        def preprocess_name(self, hf_name: str) -> str:
            return re.sub(self.HF_EXPERT_PATTERN, "block_sparse_moe.experts.", hf_name)

        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            if m := re.fullmatch(self.DT_BIAS_PATTERN, hf_name):
                gguf_to_hf_name_map[f"blk.{m['bid']}.ssm_dt.bias"] = qual_name + hf_name
            elif m := re.fullmatch(self.E_SCORE_BIAS_PATTERN, hf_name):
                gguf_to_hf_name_map[f"blk.{m['bid']}.exp_probs_b.bias"] = (
                    qual_name + hf_name
                )

        def process(self, weights, name: str, **kwargs):
            if m := re.fullmatch(self.GGUF_MOE_EXPS_PATTERN, name):
                tensor_key_mapping = kwargs.get("tensor_key_mapping")
                parsed_parameters = kwargs.get("parsed_parameters")
                if tensor_key_mapping and m["name"] in tensor_key_mapping:
                    self._split_expert_tensor(
                        weights,
                        parsed_parameters,
                        tensor_key_mapping[m["name"]],
                        m["w"],
                    )
                    return GGUFTensor(weights, None, {})
            return GGUFTensor(weights, name, {})

        def _split_expert_tensor(self, weights, parsed_parameters, hf_name, w):
            """Split fused [out, in, num_experts] tensor into individual experts."""
            w_map = {"gate": "w1", "down": "w2", "up": "w3"}
            w_name = w_map[w]
            num_experts = weights.shape[-1]
            base = hf_name.replace(".weight", "")
            for i in range(num_experts):
                expert_weights = weights[..., i]
                expert_name = base.replace(
                    "block_sparse_moe.experts.",
                    f"block_sparse_moe.experts.{i}.",
                )
                expert_name = expert_name.replace(
                    f".experts.{i}.{w_map.get(w, w)}",
                    f".experts.{i}.{w_name}",
                )
                expert_key = expert_name + ".weight"
                parsed_parameters["tensors"][expert_key] = torch.from_numpy(
                    np.copy(expert_weights)
                )

    TENSOR_PROCESSORS["kimi-linear"] = KimiLinearTensorProcessor

    orig_load = _gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "kimi-linear":
            config["model_type"] = "kimi_linear"
            config["architectures"] = ["KimiLinearForCausalLM"]
            config["auto_map"] = {
                "AutoConfig": f"{SOURCE_REPO}--configuration_kimi.KimiLinearConfig",
                "AutoModelForCausalLM": f"{SOURCE_REPO}--modeling_kimi.KimiLinearForCausalLM",
            }

            num_heads = config.get("num_attention_heads", 32)
            config["num_key_value_heads"] = num_heads
            config["mla_use_nope"] = True
            config["moe_renormalize"] = True
            config["moe_layer_freq"] = 1
            config["use_grouped_topk"] = True
            config["num_expert_group"] = 1

            rope_dim = config.pop("qk_rope_head_dim", 64)
            config["qk_rope_head_dim"] = rope_dim
            key_length_mla = config.pop("_gguf_key_length_mla", 192)
            config["qk_nope_head_dim"] = key_length_mla - rope_dim
            config.pop("_gguf_attn_key_length", None)
            config.pop("_gguf_head_count_kv", None)

            gating_func = config.pop("_gguf_gating_func", 2)
            config["moe_router_activation_func"] = (
                "sigmoid" if gating_func == 2 else "softmax"
            )

            ssm_conv_kernel = config.pop("_gguf_ssm_conv_kernel", 4)
            kda_head_dim = config.pop("_gguf_kda_head_dim", 128)
            num_layers = config.get("num_hidden_layers", 27)
            first_dense = config.get("first_k_dense_replace", 1)

            full_attn_layers = []
            kda_layers = []
            for i in range(first_dense, num_layers):
                layer_1idx = i + 1
                if layer_1idx % 4 == 0 or layer_1idx == num_layers:
                    full_attn_layers.append(layer_1idx)
                else:
                    kda_layers.append(layer_1idx)

            config["linear_attn_config"] = {
                "full_attn_layers": full_attn_layers,
                "kda_layers": kda_layers,
                "head_dim": kda_head_dim,
                "num_heads": num_heads,
                "short_conv_kernel_size": ssm_conv_kernel,
            }
        return result

    _gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "kimi_linear":
            model_type = "kimi-linear"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_kimi_linear_gguf()

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
    """Available Kimi Linear 48B A3B Instruct MXFP4 MOE GGUF model variants for causal language modeling."""

    KIMI_LINEAR_48B_A3B_INSTRUCT_MXFP4_MOE_GGUF = "48B_A3B_Instruct_MXFP4_MOE_GGUF"


class ModelLoader(ForgeModel):
    """Kimi Linear 48B A3B Instruct MXFP4 MOE GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.KIMI_LINEAR_48B_A3B_INSTRUCT_MXFP4_MOE_GGUF: LLMModelConfig(
            pretrained_model_name="noctrex/Kimi-Linear-48B-A3B-Instruct-MXFP4_MOE-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_LINEAR_48B_A3B_INSTRUCT_MXFP4_MOE_GGUF

    GGUF_FILE = "Kimi-Linear-48B-A3B-Instruct-MXFP4_MOE_BF16.gguf"

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
            model="Kimi Linear 48B A3B Instruct MXFP4 MOE GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            SOURCE_REPO, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    @staticmethod
    def _patch_model_module_fla_refs():
        """Patch fla function references in the already-imported model module."""
        import sys

        try:
            import fla.ops.kda as _kda
            import fla.ops.kda.gate as _gate
        except ImportError:
            return
        for name, mod in sys.modules.items():
            if "modeling_kimi" not in name:
                continue
            for attr, src in [
                ("chunk_kda", _kda),
                ("fused_recurrent_kda", _kda),
                ("fused_kda_gate", _gate),
            ]:
                if hasattr(mod, attr):
                    setattr(mod, attr, getattr(src, attr))

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
            config = AutoConfig.from_pretrained(SOURCE_REPO, trust_remote_code=True)
            config.num_hidden_layers = self.num_layers
            if hasattr(config, "linear_attn_config") and config.linear_attn_config:
                n = self.num_layers
                config.linear_attn_config["full_attn_layers"] = [
                    x for x in config.linear_attn_config["full_attn_layers"] if x <= n
                ]
                config.linear_attn_config["kda_layers"] = [
                    x for x in config.linear_attn_config["kda_layers"] if x <= n
                ]
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        model.config._attn_implementation = "eager"
        if hasattr(model, "model") and hasattr(model.model, "_use_flash_attention_2"):
            model.model._use_flash_attention_2 = False

        self._patch_model_module_fla_refs()
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
            if hasattr(layer, "block_sparse_moe"):
                moe = layer.block_sparse_moe
                if hasattr(moe, "shared_experts"):
                    shard_specs[moe.shared_experts.gate_proj.weight] = (
                        "model",
                        "batch",
                    )
                    shard_specs[moe.shared_experts.up_proj.weight] = (
                        "model",
                        "batch",
                    )
                    shard_specs[moe.shared_experts.down_proj.weight] = (
                        "batch",
                        "model",
                    )
            elif hasattr(layer, "mlp"):
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(SOURCE_REPO, trust_remote_code=True)
        return self.config
