# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-30B-A3B EAGLE3 speculative decoding draft model loader for causal language modeling.
"""

import json
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from transformers import AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen3-30B-A3B EAGLE3 model variants."""

    QWEN3_30B_A3B_INSTRUCT_2507_EAGLE3 = "Qwen3_30B_A3B_Instruct_2507_Eagle3"


class ModelLoader(ForgeModel):
    """Qwen3-30B-A3B EAGLE3 model loader for causal language modeling tasks."""

    BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    _VARIANTS = {
        ModelVariant.QWEN3_30B_A3B_INSTRUCT_2507_EAGLE3: ModelConfig(
            pretrained_model_name="RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_30B_A3B_INSTRUCT_2507_EAGLE3

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen3-30B-A3B-EAGLE3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL_NAME, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        config_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        config = Eagle3SpeculatorConfig(**raw_config)

        if self.num_layers is not None:
            config.transformer_layer_config.num_hidden_layers = self.num_layers

        weights_path = hf_hub_download(pretrained_model_name, "model.safetensors")
        state_dict = load_file(weights_path)

        t2d = state_dict.pop("t2d", None)
        d2t = state_dict.pop("d2t", None)

        model = Eagle3DraftModel(config, t2d=t2d, d2t=d2t)
        model.load_state_dict(state_dict, strict=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model.eval()

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        test_input = "What is the capital of France?"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
