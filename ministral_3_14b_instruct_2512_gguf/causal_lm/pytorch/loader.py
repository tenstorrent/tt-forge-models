# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral-3-14B-Instruct-2512 GGUF model loader implementation for causal language modeling.

Note: The mistral3 GGUF architecture is not yet supported by the transformers
GGUF loader, so we load from the HF-native checkpoint and extract the causal LM.
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Mistral3ForConditionalGeneration,
    Ministral3ForCausalLM,
)
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

# The HF-native checkpoint (mistral3 GGUF arch is not supported by transformers).
HF_MODEL_NAME = "unsloth/Ministral-3-14B-Instruct-2512"


class ModelVariant(StrEnum):
    """Available Ministral-3-14B-Instruct-2512 GGUF model variants for causal language modeling."""

    MINISTRAL_3_14B_INSTRUCT_2512_GGUF = "Ministral-3-14B-Instruct-2512-GGUF"


class ModelLoader(ForgeModel):
    """Ministral-3-14B-Instruct-2512 GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_14B_INSTRUCT_2512_GGUF: LLMModelConfig(
            pretrained_model_name=HF_MODEL_NAME,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_14B_INSTRUCT_2512_GGUF

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
            model="Ministral-3-14B-Instruct-2512 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
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

        # Load the full conditional generation model, then extract the causal LM
        # because the base repo uses Mistral3ForConditionalGeneration (multimodal).
        full_model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        text_config = full_model.config.text_config
        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers
        model = Ministral3ForCausalLM(text_config)
        model.model = full_model.model.language_model
        model.lm_head = full_model.lm_head
        model = model.eval()

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
        full_config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        self.config = full_config.text_config
        return self.config
