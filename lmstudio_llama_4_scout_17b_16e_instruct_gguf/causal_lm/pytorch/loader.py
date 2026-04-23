# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
lmstudio-community/Llama-4-Scout-17B-16E-Instruct-GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_transformers_llama4_gguf():
    """Monkey-patch transformers to add llama4 GGUF architecture support.

    transformers lacks the config mapping and architecture registration needed
    to load llama4 GGUF checkpoints directly.  This patch:
    - Registers llama4 in GGUF_SUPPORTED_ARCHITECTURES and config/tensor maps.
    - Remaps model_type "llama4" -> "llama4_text" so AutoConfig returns
      Llama4TextConfig (what Llama4ForCausalLM actually expects).
    - Adds GGUF_TO_FAST_CONVERTERS["llama4_text"] for tokenizer conversion.
    - Patches get_gguf_hf_weights_map to translate "llama4_text" back to
      "llama4" for gguf-py tensor name lookup.
    - Registers a LlamaTensorProcessor for llama4 Q/K weight de-permutation.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        LlamaTensorProcessor,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "llama4" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("llama4")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["llama4"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "embedding_length": "hidden_size",
        "feed_forward_length": "intermediate_size_mlp",
        "expert_feed_forward_length": "intermediate_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "interleave_moe_layer_step": "interleave_moe_layer_step",
    }

    GGUF_TO_FAST_CONVERTERS["llama4"] = GGUFLlamaConverter
    GGUF_TO_FAST_CONVERTERS["llama4_text"] = GGUFLlamaConverter
    TENSOR_PROCESSORS["llama4"] = LlamaTensorProcessor

    _orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "llama4":
            config["model_type"] = "llama4_text"
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as _tok_auto
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_utils as _modeling_utils
    import transformers.tokenization_utils_tokenizers as _tok_tokenizers

    for _mod in (_tok_auto, _config_utils, _modeling_utils, _tok_tokenizers):
        if hasattr(_mod, "load_gguf_checkpoint"):
            _mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    _orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, **kwargs
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "llama4_text":
            model_type = "llama4"
        return _orig_get_weights_map(
            hf_model, processor, model_type=model_type, **kwargs
        )

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    _modeling_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_llama4_gguf()

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
    """Available Llama 4 Scout GGUF model variants for causal language modeling."""

    LLAMA_4_SCOUT_17B_16E_INSTRUCT_GGUF = "17B_16E_Instruct_GGUF"


class ModelLoader(ForgeModel):
    """lmstudio-community Llama 4 Scout 17B 16E Instruct GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_4_SCOUT_17B_16E_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Llama-4-Scout-17B-16E-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_4_SCOUT_17B_16E_INSTRUCT_GGUF

    GGUF_FILE = "Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf"

    sample_text = "What are the main advantages of mixture-of-experts models?"

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
            model="lmstudio Llama 4 Scout 17B 16E Instruct GGUF",
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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
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

        messages = [{"role": "user", "content": self.sample_text}]
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
