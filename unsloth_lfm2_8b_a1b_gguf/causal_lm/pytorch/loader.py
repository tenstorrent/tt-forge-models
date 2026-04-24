# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth LFM2 8B A1B GGUF model loader implementation for causal language modeling.
"""
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
    TENSOR_PROCESSORS,
    Lfm2TensorProcessor,
)
from transformers.integrations.ggml import (
    GGUF_TO_FAST_CONVERTERS,
    GGUFGPTConverter,
)


class _Lfm2MoeGGUFConverter(GGUFGPTConverter):
    """GPT2-BPE converter for lfm2moe that sets bos/eos token strings from vocabulary."""

    def converted(self):
        tokenizer = super().converted()
        proto = self.original_tokenizer
        if getattr(proto, "bos_token_id", None) is not None:
            self.additional_kwargs["bos_token"] = proto.tokens[proto.bos_token_id]
        if getattr(proto, "eos_token_id", None) is not None:
            self.additional_kwargs["eos_token"] = proto.tokens[proto.eos_token_id]
        if getattr(proto, "pad_token_id", None) is not None:
            self.additional_kwargs["pad_token"] = proto.tokens[proto.pad_token_id]
        return tokenizer


from typing import Optional

from ....base import ForgeModel

_LFM2MOE_CONFIG_MAPPING = {
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
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "leading_dense_block_count": "num_dense_layers",
}


def _patch_lfm2moe_support():
    """Register lfm2moe GGUF architecture as an alias for lfm2_moe."""
    if "lfm2moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "lfm2" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                "lfm2moe"
            ] = _LFM2MOE_CONFIG_MAPPING
    # lfm2moe uses a BPE/GPT2-style tokenizer with explicit special token mapping
    GGUF_TO_FAST_CONVERTERS["lfm2_moe"] = _Lfm2MoeGGUFConverter
    GGUF_TO_FAST_CONVERTERS["lfm2moe"] = _Lfm2MoeGGUFConverter
    # lfm2moe uses the same tensor layout as lfm2
    TENSOR_PROCESSORS["lfm2moe"] = Lfm2TensorProcessor

    # Patch get_gguf_hf_weights_map to remap lfm2_moe -> lfm2moe for gguf-py lookup
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hf_model is not None:
            model_type = hf_model.config.model_type
        if model_type == "lfm2_moe":
            model_type = "lfm2moe"
        return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add lfm2moe support and fix config post-load."""
    _patch_lfm2moe_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "lfm2moe":
        result["config"]["model_type"] = "lfm2_moe"
        kv_heads = result["config"].get("num_key_value_heads")
        if isinstance(kv_heads, list):
            result["config"]["num_key_value_heads"] = max(kv_heads)
            result["config"]["layer_types"] = [
                "full_attention" if n > 0 else "conv" for n in kv_heads
            ]
        rope_theta = result["config"].pop("rope_theta", None)
        if rope_theta is not None:
            result["config"]["rope_parameters"] = {
                "rope_theta": float(rope_theta),
                "rope_type": "default",
            }
    return result


_patch_lfm2moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
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
    """Available Unsloth LFM2 8B A1B GGUF model variants for causal language modeling."""

    UNSLOTH_LFM2_8B_A1B_GGUF = "8B_A1B_GGUF"


class ModelLoader(ForgeModel):
    """Unsloth LFM2 8B A1B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.UNSLOTH_LFM2_8B_A1B_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/LFM2-8B-A1B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTH_LFM2_8B_A1B_GGUF

    GGUF_FILE = "LFM2-8B-A1B-Q4_K_M.gguf"

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
            model="Unsloth LFM2 8B A1B GGUF",
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
