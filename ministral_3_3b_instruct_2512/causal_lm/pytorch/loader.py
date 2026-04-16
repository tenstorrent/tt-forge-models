# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral-3-3B-Instruct-2512 model loader implementation for causal language modeling.
"""
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoConfig,
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


class ModelVariant(StrEnum):
    """Available Ministral-3-3B-Instruct-2512 model variants for causal language modeling."""

    MINISTRAL_3_3B_INSTRUCT_2512 = "Ministral_3_3B_Instruct_2512"


class ModelLoader(ForgeModel):
    """Ministral-3-3B-Instruct-2512 model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_3B_INSTRUCT_2512: LLMModelConfig(
            pretrained_model_name="unsloth/Ministral-3-3B-Instruct-2512",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_3B_INSTRUCT_2512

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
            model="Ministral-3-3B-Instruct-2512",
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

        # The unsloth checkpoint is stored as Mistral3ForConditionalGeneration
        # (multimodal), so we load it and extract the text-only causal LM.
        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        lm_config = full_model.model.language_model.config
        if self.num_layers is not None:
            lm_config.num_hidden_layers = self.num_layers
        causal_model = Ministral3ForCausalLM(lm_config)
        causal_model.model.load_state_dict(full_model.model.language_model.state_dict())
        causal_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
        if dtype_override is not None:
            causal_model = causal_model.to(dtype_override)
        model = causal_model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

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
            self._variant_config.pretrained_model_name
        )
        return self.config
