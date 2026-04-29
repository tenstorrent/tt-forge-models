# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Darkmere 8B GGUF model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)

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


def _patch_mistral3_gguf_support():
    """Register the mistral3 GGUF architecture and route it through ministral3.

    llama.cpp uses architecture name "mistral3" for Ministral 3 8B text-only
    models, but transformers 5.2 does not list "mistral3" in
    GGUF_SUPPORTED_ARCHITECTURES and has no GGUF_TO_FAST_CONVERTERS entry for
    it. The config key names are identical to those of "mistral", so we reuse
    that mapping. The model_type is remapped to "ministral3" downstream so
    that AutoModelForCausalLM selects Ministral3ForCausalLM (text-only)
    instead of Mistral3ForConditionalGeneration (multimodal).
    """
    if "mistral3" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "mistral" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "mistral3",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["mistral"],
            )
    # The ministral3 tokenizer uses the gpt2/tekken BPE (tokenizer_type="gpt2"
    # → is_llama_3_tokenizer=True → ByteLevel pre-tokenizer), same as llama-3.
    GGUF_TO_FAST_CONVERTERS.setdefault("mistral3", GGUFLlamaConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("ministral3", GGUFLlamaConverter)


def _patched_load_gguf_checkpoint(
    gguf_checkpoint_path, return_tensors=False, model_to_load=None
):
    """Wrap load_gguf_checkpoint to add mistral3 support and remap model_type."""
    _patch_mistral3_gguf_support()
    result = _orig_load_gguf_checkpoint(
        gguf_checkpoint_path,
        return_tensors=return_tensors,
        model_to_load=model_to_load,
    )
    if result.get("config", {}).get("model_type") == "mistral3":
        result["config"]["model_type"] = "ministral3"
    return result


def _patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
    """Wrap get_gguf_hf_weights_map to remap ministral3 → mistral3 for tensor lookup.

    The gguf-py MODEL_ARCH_NAMES table has "mistral3" but not "ministral3".
    Since Ministral3ForCausalLM has the same tensor layout as the mistral3
    GGUF arch, we remap for the lookup and restore afterward.
    """
    effective_model_type = model_type
    if effective_model_type is None and hasattr(hf_model, "config"):
        effective_model_type = hf_model.config.model_type
    if effective_model_type == "ministral3":
        effective_model_type = "mistral3"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type=effective_model_type, num_layers=num_layers, qual_name=qual_name
    )


_patch_mistral3_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Darkmere 8B GGUF model variants for causal language modeling."""

    DARKMERE_8B_V0_1_GGUF = "8B_v0.1_GGUF"


class ModelLoader(ForgeModel):
    """Darkmere 8B GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DARKMERE_8B_V0_1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Darkmere-8B-v0.1-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DARKMERE_8B_V0_1_GGUF

    GGUF_FILE = "Darkmere-8B-v0.1.i1-Q4_K_M.gguf"

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
            model="Darkmere 8B GGUF",
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
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
