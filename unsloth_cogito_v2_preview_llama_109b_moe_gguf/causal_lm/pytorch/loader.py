# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth Cogito v2 Preview Llama 109B MoE GGUF model loader implementation for causal language modeling.
"""
import torch
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


def _patch_gguf_for_llama4():
    """Patch transformers to support llama4 GGUF architecture.

    Transformers does not yet ship llama4 GGUF support. We register the config
    field mapping and fix the model_type/weight-map round-trips so that
    load_gguf_checkpoint can load this architecture.  The approach mirrors how
    transformers handles gemma3 → gemma3_text internally.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf

    if "llama4" in _gguf.GGUF_SUPPORTED_ARCHITECTURES:
        return

    _gguf.GGUF_CONFIG_MAPPING["llama4"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size_mlp",
        "expert_feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "interleave_moe_layer_step": "interleave_moe_layer_step",
    }
    _gguf.GGUF_TO_TRANSFORMERS_MAPPING["config"]["llama4"] = _gguf.GGUF_CONFIG_MAPPING[
        "llama4"
    ]
    _gguf.GGUF_SUPPORTED_ARCHITECTURES = list(
        _gguf.GGUF_TO_TRANSFORMERS_MAPPING["config"].keys()
    )

    _orig_load_gguf_checkpoint = _gguf.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        result = _orig_load_gguf_checkpoint(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )
        if result.get("config", {}).get("model_type") == "llama4":
            result["config"]["model_type"] = "llama4_text"
        return result

    _gguf.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    _orig_get_gguf_hf_weights_map = _gguf.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "llama4_text":
            model_type = "llama4"
        return _orig_get_gguf_hf_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    _gguf.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_gguf_for_llama4()


class ModelVariant(StrEnum):
    """Available Unsloth Cogito v2 Preview Llama 109B MoE GGUF model variants for causal language modeling."""

    COGITO_V2_PREVIEW_LLAMA_109B_MOE_Q4_K_M = "Cogito_V2_Preview_Llama_109B_MoE_Q4_K_M"


class ModelLoader(ForgeModel):
    """Unsloth Cogito v2 Preview Llama 109B MoE GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.COGITO_V2_PREVIEW_LLAMA_109B_MOE_Q4_K_M: LLMModelConfig(
            pretrained_model_name="unsloth/cogito-v2-preview-llama-109B-MoE-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COGITO_V2_PREVIEW_LLAMA_109B_MOE_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.COGITO_V2_PREVIEW_LLAMA_109B_MOE_Q4_K_M: "Q4_K_M/cogito-v2-preview-llama-109B-MoE-Q4_K_M-00001-of-00002.gguf",
    }

    sample_text = "What are the main advantages of mixture-of-experts models?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers
        self.gguf_file = self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Unsloth Cogito v2 Preview Llama 109B MoE GGUF",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
