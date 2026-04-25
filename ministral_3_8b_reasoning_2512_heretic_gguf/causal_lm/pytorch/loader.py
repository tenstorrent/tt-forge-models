# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
coder3101 Ministral-3-8B-Reasoning-2512-heretic GGUF model loader for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel


def _apply_gguf_patches():
    """Register mistral3 GGUF support in transformers.

    The GGUF architecture name 'mistral3' (used by Ministral 3 models) is not
    recognized by transformers' GGUF loader. This patch maps it to MistralConfig
    and MistralForCausalLM, which share the same tensor layout.
    Safe to call multiple times; idempotent thanks to the _PATCHED sentinel.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    if getattr(_gguf_utils, "_mistral3_patched", False):
        return

    from transformers import MistralConfig
    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    if "mistral3" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["mistral3"] = GGUF_CONFIG_MAPPING["mistral"].copy()

    if "mistral3" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")

    if "mistral3" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["mistral3"] = GGUFGPTConverter

    if "mistral3" not in CONFIG_MAPPING._extra_content:
        CONFIG_MAPPING._extra_content["mistral3"] = MistralConfig

    if "mistral3" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["mistral3"] = "MistralForCausalLM"

    _gguf_utils._mistral3_patched = True


from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


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
        _apply_gguf_patches()
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
        _apply_gguf_patches()
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
        _apply_gguf_patches()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
