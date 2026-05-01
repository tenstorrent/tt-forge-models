# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral 8B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel


def _patch_transformers_mistral3_gguf():
    """Monkey-patch transformers to add mistral3 GGUF architecture support.

    Transformers 5.x lacks GGUF loading for the mistral3 architecture used by
    Ministral models. mistral3 shares the same tensor layout as mistral, so we
    register the architecture, remap model_type → mistral, and explicitly set
    sliding_window=None so that MistralForCausalLM uses plain causal attention.
    Without sliding_window=None the default MistralConfig.sliding_window=4096
    would trigger SlidingWindowLayer with out-of-range negative slice indices.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "mistral3" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["mistral3"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "mistral3" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["mistral3"] = GGUFLlamaConverter
    # The GGUF tokenizer.ggml.model field is "mistral" for this model family;
    # convert_gguf_tokenizer looks up by that key.
    if "mistral" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["mistral"] = GGUFLlamaConverter

    _orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "mistral3":
            # Remap to mistral and set sliding_window=None so AutoModelForCausalLM
            # uses MistralForCausalLM with full (non-windowed) causal attention.
            # Using the default MistralConfig sets sliding_window=4096 which
            # incorrectly limits attention on short sequences → PCC ~0.95.
            config["model_type"] = "mistral"
            config["sliding_window"] = None
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # configuration_utils.py and tokenization modules do a module-level
    # `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint`, binding
    # the original function before our patch runs. Patch those bindings too.
    import transformers.configuration_utils as _config_utils
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    import transformers.tokenization_utils_tokenizers as _tok_utils
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    import transformers.models.auto.tokenization_auto as _tok_auto
    _tok_auto.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # get_gguf_hf_weights_map uses hf_model.config.model_type ("ministral3") to
    # look up the arch in gguf-py's MODEL_ARCH_NAMES, but gguf-py 0.10+ only has
    # "mistral3" (not "ministral3"). Patch the function to remap before the lookup.
    _orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        effective_type = hf_model.config.model_type if model_type is None else model_type
        if effective_type == "mistral":
            model_type = "mistral3"
        return _orig_get_weights_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_mistral3_gguf()
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
    """Available Ministral 8B GGUF model variants for causal language modeling."""

    MINISTRAL_8B_INSTRUCT_2512_GGUF = "8B_Instruct_2512_GGUF"
    LMSTUDIO_MINISTRAL_3_8B_INSTRUCT_2512_GGUF = (
        "lmstudio_Ministral-3-8B-Instruct-2512-GGUF"
    )


class ModelLoader(ForgeModel):
    """Ministral 8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_8B_INSTRUCT_2512_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Ministral-3-8B-Instruct-2512-GGUF",
            max_length=128,
        ),
        ModelVariant.LMSTUDIO_MINISTRAL_3_8B_INSTRUCT_2512_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Ministral-3-8B-Instruct-2512-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_8B_INSTRUCT_2512_GGUF

    GGUF_FILE = "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"

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
            model="Ministral 8B GGUF",
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
