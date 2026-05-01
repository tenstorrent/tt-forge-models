# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui LFM2 24B A2B Abliterated i1-GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional


def _find_base_load_gguf():
    """Find the original transformers load_gguf_checkpoint function.

    Many loaders patch gguf_utils.load_gguf_checkpoint at import time using
    signatures that omit **kwargs. Search sys.modules for the real function
    (identified by __module__ and __qualname__) to bypass broken wrappers.
    """
    import sys
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    _target_mod = "transformers.modeling_gguf_pytorch_utils"
    _target_name = "load_gguf_checkpoint"

    def _is_original(fn):
        return (
            getattr(fn, "__module__", "") == _target_mod
            and getattr(fn, "__qualname__", "") == _target_name
        )

    if _is_original(gguf_utils.load_gguf_checkpoint):
        return gguf_utils.load_gguf_checkpoint

    for mod in list(sys.modules.values()):
        mod_dict = getattr(mod, "__dict__", None)
        if not mod_dict:
            continue
        for val in mod_dict.values():
            if callable(val) and _is_original(val):
                return val

    return gguf_utils.load_gguf_checkpoint


def _register_lfm2moe_gguf_arch():
    """Register lfm2moe in transformers' GGUF architecture tables.

    LFM2 24B A2B uses GGUF architecture 'lfm2moe' but transformers 5.2.x
    only registers 'lfm2' (dense variant). The HF model_type is 'lfm2_moe'.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )

    if "lfm2moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")

    if "lfm2moe" not in TENSOR_PROCESSORS and "lfm2" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["lfm2moe"] = TENSOR_PROCESSORS["lfm2"]

    if "lfm2moe" not in GGUF_TO_TRANSFORMERS_MAPPING["config"]:
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["lfm2moe"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "norm_eps",
            "vocab_size": "vocab_size",
            "shortconv.l_cache": "conv_L_cache",
            "expert_count": "num_experts",
            "expert_feed_forward_length": "moe_intermediate_size",
            "expert_used_count": "num_experts_per_tok",
            "leading_dense_block_count": "num_dense_layers",
            "expert_gating_func": None,
        }

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    if "lfm2moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["lfm2moe"] = GGUFGPTConverter
    if "lfm2_moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["lfm2_moe"] = GGUFGPTConverter


def _patch_grouped_mm_experts_forward():
    """Fix grouped_mm_experts_forward to use float histc input on non-CUDA devices.

    transformers uses expert_ids_g.int() for non-CPU devices but torch.histc on
    CPU (where XLA ops fall back to) only supports float input. When device.type
    is "xla" the int path is chosen, causing:
        NotImplementedError: "histogram_cpu" not implemented for 'Int'
    Fix: use float whenever device.type != "cuda".
    """
    import transformers.integrations.moe as moe_module

    if getattr(moe_module, "_lfm2moe_histc_patched", False):
        return

    _grouped_linear = moe_module._grouped_linear

    def _patched_gmef(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden_states.device
        num_top_k = top_k_index.size(-1)
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)

        token_idx = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1)
            .expand(-1, num_top_k)
            .reshape(-1)
        )
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)

        selected_hidden_states = hidden_states[token_idx]

        perm = torch.argsort(expert_ids)
        inv_perm = torch.argsort(perm)
        expert_ids_g = expert_ids[perm]
        sample_weights_g = sample_weights[perm]
        selected_hidden_states_g = selected_hidden_states[perm]

        selected_gate_up = self.gate_up_proj
        selected_down = self.down_proj
        selected_gate_up_bias = (
            self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
        )
        selected_down_bias = (
            self.down_proj_bias[expert_ids_g] if self.has_bias else None
        )

        # CUDA supports int histc; CPU and XLA (which falls back to CPU) require float.
        histc_input = (
            expert_ids_g.float() if device.type != "cuda" else expert_ids_g.int()
        )
        num_tokens_per_expert = torch.histc(
            histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1
        )
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        gate_up_out = _grouped_linear(
            selected_hidden_states_g,
            selected_gate_up,
            selected_gate_up_bias,
            offsets,
            is_transposed=self.is_transposed,
        )
        gated_out = self._apply_gate(gate_up_out)
        out_per_sample_g = _grouped_linear(
            gated_out,
            selected_down,
            selected_down_bias,
            offsets,
            is_transposed=self.is_transposed,
        )
        out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)
        out_per_sample = out_per_sample_g[inv_perm]
        final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(
            dim=1
        )
        return final_hidden_states.to(hidden_states.dtype)

    moe_module.grouped_mm_experts_forward = _patched_gmef
    moe_module.ExpertsInterface._global_mapping["grouped_mm"] = _patched_gmef
    moe_module._lfm2moe_histc_patched = True


def _apply_lfm2moe_load_patches():
    """Wrap load_gguf_checkpoint for lfm2moe model_type remapping.

    Uses _find_base_load_gguf() to bypass broken intermediate wrappers that
    omit **kwargs, then patches all import sites so the wrapper is always
    outermost when from_pretrained calls load_gguf_checkpoint.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.tokenization_utils_tokenizers as tok_tokenizers

    _patch_grouped_mm_experts_forward()

    if getattr(gguf_utils.load_gguf_checkpoint, "_is_lfm2moe_wrapper", False):
        return

    base_load = _find_base_load_gguf()

    def _lfm2moe_load(gguf_checkpoint_path, return_tensors=False, **kwargs):
        result = base_load(gguf_checkpoint_path, return_tensors=return_tensors, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "lfm2moe":
            config["model_type"] = "lfm2_moe"
            gguf_kv_heads = config.get("num_key_value_heads", [])
            if isinstance(gguf_kv_heads, list) and gguf_kv_heads:
                full_attn_idxs = [i for i, n in enumerate(gguf_kv_heads) if n > 0]
                config["num_key_value_heads"] = max(gguf_kv_heads)
                config["block_auto_adjust_ff_dim"] = False
                n_layers = config.get("num_hidden_layers", len(gguf_kv_heads))
                config["layer_types"] = [
                    "full_attention" if i in full_attn_idxs else "short_conv"
                    for i in range(n_layers)
                ]
        return result

    _lfm2moe_load._is_lfm2moe_wrapper = True

    gguf_utils.load_gguf_checkpoint = _lfm2moe_load
    for mod in (tok_auto, config_utils, modeling_utils, tok_tokenizers):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _lfm2moe_load

    prev_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _lfm2moe_weights_map(
        hf_model, processor=None, model_type=None, num_layers=None, qual_name=""
    ):
        mt = hf_model.config.model_type if model_type is None else model_type
        if mt == "lfm2_moe":
            mt = "lfm2moe"
        return prev_weights_map(
            hf_model, processor, model_type=mt, num_layers=num_layers, qual_name=qual_name
        )

    gguf_utils.get_gguf_hf_weights_map = _lfm2moe_weights_map


# Register at import time so AutoConfig works during test collection.
_register_lfm2moe_gguf_arch()

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
    """Available Huihui LFM2 24B A2B Abliterated i1-GGUF model variants for causal language modeling."""

    HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF = "24B_A2B_i1_GGUF"
    HUIHUI_LFM2_24B_A2B_ABLITERATED_GGUF = "24B_A2B_GGUF"


class ModelLoader(ForgeModel):
    """Huihui LFM2 24B A2B Abliterated i1-GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-LFM2-24B-A2B-abliterated-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-LFM2-24B-A2B-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF: "Huihui-LFM2-24B-A2B-abliterated.i1-Q4_K_M.gguf",
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_GGUF: "Huihui-LFM2-24B-A2B-abliterated.Q4_K_M.gguf",
    }

    sample_text = "The quick brown fox jumps over the lazy dog."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Huihui LFM2 24B A2B Abliterated i1-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _apply_lfm2moe_load_patches()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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

        # Re-apply just before from_pretrained so our wrapper is outermost.
        _apply_lfm2moe_load_patches()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if self.tokenizer is not None and len(self.tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))

        model.eval()
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

    def load_config(self):
        _apply_lfm2moe_load_patches()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
