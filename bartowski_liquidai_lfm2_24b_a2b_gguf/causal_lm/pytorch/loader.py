# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski LiquidAI LFM2-24B-A2B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if the package was installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


# GGUF config key -> transformers config key mapping for lfm2moe architecture
_LFM2MOE_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
    "shortconv.l_cache": "conv_L_cache",
    "expert_count": "num_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "leading_dense_block_count": "num_dense_layers",
}


def _patch_lfm2moe_gguf_support():
    """Patch transformers to support lfm2moe GGUF architecture (maps to lfm2_moe model type).

    The transformers library supports lfm2_moe as a model type, but the GGUF
    architecture name lfm2moe (no underscore) is missing from its GGUF loading
    infrastructure. This function adds the necessary mappings.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.tokenization_utils_tokenizers as _tok_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer

    if "lfm2moe" in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        return

    from transformers.modeling_gguf_pytorch_utils import Lfm2TensorProcessor

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "lfm2moe"
    ] = _LFM2MOE_CONFIG_MAPPING
    _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")
    _gguf_utils.TENSOR_PROCESSORS["lfm2moe"] = Lfm2TensorProcessor

    # Patch get_gguf_hf_weights_map to handle lfm2_moe model_type -> lfm2moe gguf arch
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        resolved_type = (
            model_type
            if model_type is not None
            else getattr(hf_model.config, "model_type", None)
        )
        if resolved_type == "lfm2_moe":
            model_type = "lfm2moe"
        return _orig_get_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map

    # Wrap load_gguf_checkpoint to fix lfm2moe config after loading
    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load(gguf_path, return_tensors=False, **kwargs):
        result = _orig_load(gguf_path, return_tensors=return_tensors, **kwargs)
        if result.get("config", {}).get("model_type") == "lfm2moe":
            kv_heads = result["config"].get("num_key_value_heads", 8)
            if isinstance(kv_heads, list):
                kv_heads = max(kv_heads) if max(kv_heads) > 0 else 8
            result["config"]["num_key_value_heads"] = kv_heads
            result["config"]["model_type"] = "lfm2_moe"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load
    _config_utils.load_gguf_checkpoint = _patched_load
    _tok_utils.load_gguf_checkpoint = _patched_load
    _auto_tokenizer.load_gguf_checkpoint = _patched_load


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
    """Available bartowski LiquidAI LFM2-24B-A2B GGUF model variants for causal language modeling."""

    BARTOWSKI_LIQUIDAI_LFM2_24B_A2B_GGUF = "LiquidAI_LFM2_24B_A2B_GGUF"


class ModelLoader(ForgeModel):
    """bartowski LiquidAI LFM2-24B-A2B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_LIQUIDAI_LFM2_24B_A2B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/LiquidAI_LFM2-24B-A2B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_LIQUIDAI_LFM2_24B_A2B_GGUF

    GGUF_FILE = "LiquidAI_LFM2-24B-A2B-Q4_K_M.gguf"

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
            model="bartowski LiquidAI LFM2-24B-A2B GGUF",
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
        _refresh_gguf_detection()
        _patch_lfm2moe_gguf_support()
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
        _refresh_gguf_detection()
        _patch_lfm2moe_gguf_support()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
