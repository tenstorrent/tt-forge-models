# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF model loader for causal language modeling.
"""

import os
import re
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

_MOONSHOTAI_REPO = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
_MRADERMACHER_REPO = "mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF"
_GGUF_FILENAME = "Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated.Q4_K_M.gguf"

_kimi_patched = False


def _patch_transformers_kimi_linear_gguf():
    global _kimi_patched
    if _kimi_patched:
        return
    _kimi_patched = True

    # modeling_kimi.py imports OutputRecorder and check_model_inputs from
    # transformers.utils.generic, but this version of transformers has them elsewhere.
    import transformers.utils.generic as _tug

    if not hasattr(_tug, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder as _OR

        _tug.OutputRecorder = _OR

    if not hasattr(_tug, "check_model_inputs"):
        _tug.check_model_inputs = lambda fn: fn

    # modeling_kimi.py unconditionally forces flash_attention_2 even when eager is
    # requested. Patch the transformers base class so that when flash_attn is not
    # available, the attn implementation gracefully falls back to eager instead of
    # raising ImportError.
    import transformers.modeling_utils as _mu

    _orig_check_adjust = _mu.PreTrainedModel._check_and_adjust_attn_implementation

    def _patched_check_and_adjust(self, attn_implementation, is_init_check=False):
        try:
            return _orig_check_adjust(self, attn_implementation, is_init_check)
        except ImportError:
            return "eager"

    _mu.PreTrainedModel._check_and_adjust_attn_implementation = (
        _patched_check_and_adjust
    )

    import inspect

    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    # Other loaders patch load_gguf_checkpoint without accepting torch_dtype (e.g.
    # bartowski_coniccat_qwen3_5_27b_writer_gguf). Wrap it so torch_dtype is accepted
    # and tensors are cast after loading when needed.
    if (
        "torch_dtype"
        not in inspect.signature(gguf_utils.load_gguf_checkpoint).parameters
    ):
        _wrapped_load = gguf_utils.load_gguf_checkpoint

        def _load_gguf_with_dtype(
            path, return_tensors=False, model_to_load=None, torch_dtype=None, **kw
        ):
            result = _wrapped_load(
                path, return_tensors=return_tensors, model_to_load=model_to_load, **kw
            )
            if torch_dtype is not None and isinstance(result.get("tensors"), dict):
                result["tensors"] = {
                    k: v.to(torch_dtype) if isinstance(v, torch.Tensor) else v
                    for k, v in result["tensors"].items()
                }
            return result

        gguf_utils.load_gguf_checkpoint = _load_gguf_with_dtype
    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    from transformers.modeling_gguf_pytorch_utils import GGUFTensor, TensorProcessor

    if "kimi-linear" not in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("kimi-linear")

    if "kimi-linear" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["kimi-linear"] = {
            "block_count": "num_hidden_layers",
            "context_length": "max_position_embeddings",
            "embedding_length": "hidden_size",
            "feed_forward_length": "intermediate_size",
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": None,
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
            "expert_count": "num_experts",
            "expert_used_count": "num_experts_per_token",
            "expert_feed_forward_length": "moe_intermediate_size",
            "expert_shared_count": "num_shared_experts",
            "leading_dense_block_count": "first_k_dense_replace",
            "attention.kv_lora_rank": "kv_lora_rank",
            "attention.value_length_mla": "v_head_dim",
            "kda.head_dim": "qk_nope_head_dim",
            "expert_weights_scale": "routed_scaling_factor",
            "rope.dimension_count": None,
        }

    if "kimi-linear" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["kimi-linear"] = GGUFQwen2Converter

    class KimiLinearTensorProcessor(TensorProcessor):
        _SSM_A_PATTERN = re.compile(r"blk\.(\d+)\.ssm_a$")
        _K_B_PATTERN = re.compile(r"blk\.(\d+)\.attn_k_b\.weight")
        _V_B_PATTERN = re.compile(r"blk\.(\d+)\.attn_v_b\.weight")
        _MOE_EXPS_PATTERN = re.compile(
            r"blk\.(\d+)\.(ffn_gate_exps|ffn_down_exps|ffn_up_exps)\.weight"
        )

        def __init__(self, config=None):
            super().__init__(config=config)
            self._pending_k_b = {}

        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            # dt_bias is a plain Parameter: model.layers.{bid}.self_attn.dt_bias
            # → blk.{bid}.ssm_dt.bias (no mapping via name_map)
            if m := re.search(r"layers\.(\d+)\.self_attn\.dt_bias$", hf_name):
                bid = m.group(1)
                gguf_to_hf_name_map[f"blk.{bid}.ssm_dt.bias"] = qual_name + hf_name

        def process(self, weights, name, **kwargs):
            # A_log is stored as (num_heads, 1) but model expects (1, 1, num_heads, 1)
            if self._SSM_A_PATTERN.match(name):
                weights = weights.reshape(1, 1, weights.shape[0], 1)
                return GGUFTensor(weights, name, {})

            # kv_b_proj is split in GGUF into attn_k_b + attn_v_b; merge them
            if m := self._K_B_PATTERN.match(name):
                self._pending_k_b[m.group(1)] = weights
                return GGUFTensor(weights, None, {})

            if m := self._V_B_PATTERN.match(name):
                bid = m.group(1)
                tensor_key_mapping = kwargs.get("tensor_key_mapping", {})
                parsed_parameters = kwargs.get("parsed_parameters", {})
                k_b = self._pending_k_b.pop(bid, None)
                if k_b is not None:
                    # k_b: (num_heads, kv_lora_rank, qk_nope) → (num_heads, qk_nope, kv_lora_rank)
                    k_b_w = k_b.transpose(0, 2, 1)
                    # v_b: (num_heads, v_head_dim, kv_lora_rank) – already correct
                    v_b_w = weights
                    # stack: (num_heads, qk_nope + v_head_dim, kv_lora_rank)
                    kv_b = np.concatenate([k_b_w, v_b_w], axis=1)
                    kv_b_weight = kv_b.reshape(-1, kv_b.shape[-1])
                    kv_b_hf = tensor_key_mapping.get(f"blk.{bid}.attn_kv_b.weight")
                    if kv_b_hf and "tensors" in parsed_parameters:
                        parsed_parameters["tensors"][kv_b_hf] = torch.from_numpy(
                            np.copy(kv_b_weight)
                        )
                return GGUFTensor(weights, None, {})

            # Per-expert MoE weights are stacked in GGUF; split into individual experts
            if m := self._MOE_EXPS_PATTERN.match(name):
                bid = m.group(1)
                exps_type = m.group(2)
                parsed_parameters = kwargs.get("parsed_parameters", {})
                w_name = {
                    "ffn_gate_exps": "w1",
                    "ffn_down_exps": "w2",
                    "ffn_up_exps": "w3",
                }[exps_type]
                if "tensors" in parsed_parameters:
                    for i in range(weights.shape[0]):
                        hf_name = (
                            f"model.layers.{bid}.block_sparse_moe"
                            f".experts.{i}.{w_name}.weight"
                        )
                        parsed_parameters["tensors"][hf_name] = torch.from_numpy(
                            np.copy(weights[i])
                        )
                return GGUFTensor(weights, None, {})

            return GGUFTensor(weights, name, {})

    if "kimi-linear" not in gguf_utils.TENSOR_PROCESSORS:
        gguf_utils.TENSOR_PROCESSORS["kimi-linear"] = KimiLinearTensorProcessor

    # Patch get_gguf_hf_weights_map to translate kimi_linear → kimi-linear
    _orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = getattr(hf_model.config, "model_type", None)
        if model_type == "kimi_linear":
            model_type = "kimi-linear"
        return _orig_get_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = _patched_get_map


def _get_moonshotai_dir():
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        _MOONSHOTAI_REPO, "config.json", local_files_only=True
    )
    return os.path.dirname(config_path)


def _get_gguf_path():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_MRADERMACHER_REPO, _GGUF_FILENAME, local_files_only=True)


def _load_kimi_tokenizer(gguf_path):
    from transformers.integrations.ggml import convert_gguf_tokenizer
    from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

    parsed = load_gguf_checkpoint(gguf_path, return_tensors=False)
    tokenizer_dict = parsed.get("tokenizer", {})
    fast_tokenizer, additional_kwargs = convert_gguf_tokenizer(
        "kimi-linear", tokenizer_dict
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast_tokenizer, **additional_kwargs
    )
    # Prefer <|im_end|> as EOS (ChatML-style generation stopping)
    im_end_id = fast_tokenizer.token_to_id("<|im_end|>")
    if im_end_id is not None:
        tokenizer.eos_token = fast_tokenizer.id_to_token(im_end_id)
        tokenizer.eos_token_id = im_end_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class ModelVariant(StrEnum):
    """Available mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF model variants."""

    HUIHUI_KIMI_LINEAR_48B_A3B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF = (
        "48B_A3B_Instruct_Abliterated_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.HUIHUI_KIMI_LINEAR_48B_A3B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name=_MRADERMACHER_REPO,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.HUIHUI_KIMI_LINEAR_48B_A3B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF
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
            model="mradermacher/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_transformers_kimi_linear_gguf()
        gguf_path = _get_gguf_path()
        self.tokenizer = _load_kimi_tokenizer(gguf_path)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _patch_transformers_kimi_linear_gguf()

        moonshotai_dir = _get_moonshotai_dir()
        gguf_path = _get_gguf_path()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load config WITHOUT gguf_file so auto_map and linear_attn_config are preserved.
        # Passing gguf_file to AutoConfig.from_pretrained causes _get_config_dict to
        # replace config.json with GGUF-derived config, losing auto_map entirely.
        kimi_config = AutoConfig.from_pretrained(moonshotai_dir, trust_remote_code=True)
        if self.num_layers is not None:
            kimi_config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = kimi_config

        # Use model class directly (PreTrainedModel.from_pretrained) so that:
        # 1. The explicit PreTrainedConfig is preserved with linear_attn_config
        # 2. GGUF weights are still loaded via gguf_file kwarg
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        ModelClass = get_class_from_dynamic_module(
            "modeling_kimi.KimiLinearForCausalLM", moonshotai_dir
        )
        model = ModelClass.from_pretrained(
            moonshotai_dir,
            gguf_file=gguf_path,
            **model_kwargs,
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text

        inputs = self.tokenizer(
            [text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            if hasattr(layer, "block_sparse_moe"):
                moe = layer.block_sparse_moe
                if hasattr(moe, "experts"):
                    for expert in moe.experts:
                        shard_specs[expert.w1.weight] = ("model", "batch")
                        shard_specs[expert.w2.weight] = ("batch", "model")
                        shard_specs[expert.w3.weight] = ("model", "batch")
                if hasattr(moe, "shared_experts"):
                    shard_specs[moe.shared_experts.w1.weight] = ("model", "batch")
                    shard_specs[moe.shared_experts.w2.weight] = ("batch", "model")
                    shard_specs[moe.shared_experts.w3.weight] = ("model", "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        _patch_transformers_kimi_linear_gguf()
        moonshotai_dir = _get_moonshotai_dir()
        self.config = AutoConfig.from_pretrained(moonshotai_dir, trust_remote_code=True)
        return self.config
