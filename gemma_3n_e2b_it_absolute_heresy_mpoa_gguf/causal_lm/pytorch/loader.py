# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 3n E2B IT Absolute Heresy MPOA GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_gemma3n_gguf():
    """Monkey-patch transformers to add gemma3n GGUF architecture support.

    Transformers 5.x has Gemma3nForCausalLM but the GGUF loading utilities
    lack the config mapping and architecture registration for gemma3n.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "gemma3n" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("gemma3n")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["gemma3n"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "embedding_length_per_layer_input": "hidden_size_per_layer_input",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "attention.sliding_window": "sliding_window",
        "attention.shared_kv_layers": "num_kv_shared_layers",
        "altup.active_idx": "altup_active_idx",
        "altup.num_inputs": "altup_num_inputs",
        "final_logit_softcapping": "final_logit_softcapping",
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGemmaConverter,
    )

    if "gemma3n" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gemma3n"] = GGUFGemmaConverter
    if "gemma3n_text" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gemma3n_text"] = GGUFGemmaConverter

    _orig_load_gguf_checkpoint = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load_gguf_checkpoint(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "gemma3n":
            config["model_type"] = "gemma3n_text"
            # gemma3n GGUF has no vocab_size_per_layer_input metadata field;
            # the GGUF embed_tokens_per_layer uses the same vocab size as the
            # main embedding, so align vocab_size_per_layer_input with vocab_size.
            vocab_size = config.get("vocab_size")
            if vocab_size is not None:
                config["vocab_size_per_layer_input"] = vocab_size
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("gemma3n_text", "gemma3n"):
            model_type = "gemma3n"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_gemma3n_gguf()


class ModelVariant(StrEnum):
    """Available Gemma 3n E2B IT Absolute Heresy MPOA GGUF model variants for causal language modeling."""

    GEMMA_3N_E2B_IT_ABSOLUTE_HERESY_MPOA_Q4_K_M_GGUF = (
        "Gemma_3n_E2B_IT_absolute_heresy_MPOA_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Gemma 3n E2B IT Absolute Heresy MPOA GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3N_E2B_IT_ABSOLUTE_HERESY_MPOA_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/gemma-3n-E2B-it-absolute-heresy-MPOA-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3N_E2B_IT_ABSOLUTE_HERESY_MPOA_Q4_K_M_GGUF

    GGUF_FILE = "gemma-3n-E2B-it-absolute-heresy-MPOA.Q4_K_M.gguf"

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
            model="Gemma 3n E2B IT Absolute Heresy MPOA GGUF",
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
