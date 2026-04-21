# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLAO-Qwen3-8B-3L-3N model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MLAO-Qwen3-8B-3L-3N model variants."""

    MLAO_QWEN3_8B_3L_3N = "MLAO-Qwen3-8B-3L-3N"


class ModelLoader(ForgeModel):
    """MLAO-Qwen3-8B-3L-3N model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MLAO_QWEN3_8B_3L_3N: ModelConfig(
            pretrained_model_name="nluick/MLAO-Qwen3-8B-3L-3N",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MLAO_QWEN3_8B_3L_3N

    BASE_MODEL_NAME = "Qwen/Qwen3-8B"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MLAO-Qwen3-8B-3L-3N",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"padding_side": "left"}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL_NAME, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        test_input = "Explain the feature represented by this SAE activation."

        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            test_input,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
