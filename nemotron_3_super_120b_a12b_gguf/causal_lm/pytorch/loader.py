# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.

The nemotron_h_moe GGUF architecture is not yet supported by the transformers
GGUF loader, so this module loads the model config from the upstream BF16
checkpoint and instantiates the model from that config with a reduced layer
count for compile-only testing.
"""
import torch
from transformers import AutoTokenizer, AutoConfig, NemotronHForCausalLM
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


BF16_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


class ModelVariant(StrEnum):
    """Available NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_GGUF = "3_Super_120B_A12B_GGUF"


class ModelLoader(ForgeModel):
    """NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_GGUF

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
            model="Nemotron 3 Super 120B A12B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(BF16_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(BF16_MODEL_NAME)

        if self.num_layers is not None:
            config.layers_block_type = config.layers_block_type[: self.num_layers]
        else:
            config.layers_block_type = ["mamba", "moe", "attention", "moe"]
        config.num_nextn_predict_layers = 0
        config.n_routed_experts = 8
        config.num_experts_per_tok = 2

        model = NemotronHForCausalLM(config).to(
            dtype=dtype_override if dtype_override is not None else torch.bfloat16
        )
        model.eval()

        self.config = config
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

    def load_config(self):
        if self.config is not None:
            return self.config
        self.config = AutoConfig.from_pretrained(BF16_MODEL_NAME)
        return self.config
