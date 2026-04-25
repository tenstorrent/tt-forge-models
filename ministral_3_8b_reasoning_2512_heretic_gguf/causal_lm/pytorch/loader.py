# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
coder3101 Ministral-3-8B-Reasoning-2512-heretic GGUF model loader for causal language modeling.
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


def _patch_mistral3_support():
    """Register mistral3 GGUF architecture as an alias for ministral3/mistral.

    The Ministral 8B family uses 'mistral3' as the GGUF architecture tag, but
    transformers 5.x only supports 'mistral' in its GGUF loader. We register
    'mistral3' using the same config-key mapping as 'mistral' so that the GGUF
    weights can be loaded, then fix the resulting model_type to 'ministral3'
    so that AutoModelForCausalLM resolves to Ministral3ForCausalLM.
    """
    if "mistral3" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "mistral" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "mistral3",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["mistral"],
            )
    GGUF_TO_FAST_CONVERTERS.setdefault("mistral3", GGUFLlamaConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("ministral3", GGUFLlamaConverter)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add mistral3 support and fix model_type."""
    _patch_mistral3_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "mistral3":
        result["config"]["model_type"] = "ministral3"
    return result


def _patch_ministral3_weights_map():
    """Remap ministral3 -> mistral3 in get_gguf_hf_weights_map.

    After model_type is remapped from mistral3 to ministral3, the model config
    carries model_type='ministral3'. But gguf-py only knows 'mistral3', so we
    intercept the weights-map call and restore the gguf-py name.
    """
    _sentinel = "_ministral3_remap_patched"
    orig = _gguf_utils.get_gguf_hf_weights_map
    if getattr(orig, _sentinel, False):
        return

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type == "ministral3":
            model_type = "mistral3"
        return orig(hf_model, processor, model_type, num_layers, qual_name)

    setattr(_patched_get_gguf_hf_weights_map, _sentinel, True)
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_mistral3_support()
_patch_ministral3_weights_map()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Ministral-3-8B-Reasoning-2512-heretic GGUF model variants."""

    MINISTRAL_3_8B_REASONING_2512_HERETIC_Q4_K_M_GGUF = (
        "3-8B-Reasoning-2512-heretic-Q4_K_M-GGUF"
    )


class ModelLoader(ForgeModel):
    """coder3101 Ministral-3-8B-Reasoning-2512-heretic GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_8B_REASONING_2512_HERETIC_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="coder3101/Ministral-3-8B-Reasoning-2512-heretic-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_8B_REASONING_2512_HERETIC_Q4_K_M_GGUF

    GGUF_FILE = "Ministral-3-8B-Reasoning-2512-heretic-Q4_K_M.gguf"

    sample_text = "What is the capital of France?"

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
            model="coder3101 Ministral-3-8B-Reasoning-2512-heretic GGUF",
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
