# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test GGUF MoE Sample model loader implementation for causal language modeling.

The upstream GGUF repo (SzymonOzog/test-gguf-moe-sample) contains only random
quantized tensors with no config or tokenizer metadata, so the model is built
from a small Mixtral config with a standard LLaMA tokenizer.
"""
import torch
from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM
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
    """Available Test GGUF MoE Sample model variants for causal language modeling."""

    TEST_GGUF_MOE_SAMPLE_Q4_0 = "Q4_0"


class ModelLoader(ForgeModel):
    """Test GGUF MoE Sample model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TEST_GGUF_MOE_SAMPLE_Q4_0: LLMModelConfig(
            pretrained_model_name="SzymonOzog/test-gguf-moe-sample",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEST_GGUF_MOE_SAMPLE_Q4_0

    GGUF_FILE = "Quant_Q4_0_512.gguf"
    TOKENIZER_SOURCE = "NousResearch/Llama-2-7b-hf"

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
            model="Test GGUF MoE Sample",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _create_moe_config(self):
        num_layers = self.num_layers if self.num_layers is not None else 2
        return MixtralConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=32000,
            max_position_embeddings=self._variant_config.max_length,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_SOURCE)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self._create_moe_config()

        model = MixtralForCausalLM(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model = model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
        self.config = self._create_moe_config()
        return self.config
