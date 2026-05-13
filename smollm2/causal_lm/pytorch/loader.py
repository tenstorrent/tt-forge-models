# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolLM2 model loader implementation for causal language modeling.
"""
from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


class ModelVariant(StrEnum):
    """Available SmolLM2 model variants."""

    SMOLLM2_135M = "135M"
    SMOLLM2_135M_INSTRUCT = "135M_Instruct"
    SMOLLM2_360M = "360M"
    SMOLLM2_360M_INSTRUCT = "360M_Instruct"
    SMOLLM2_1_7B = "1.7B"
    SMOLLM2_1_7B_INSTRUCT = "1.7B_Instruct"


class ModelLoader(ForgeModel):
    """SmolLM2 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SMOLLM2_135M: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-135M",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_135M_INSTRUCT: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_360M: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-360M",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_360M_INSTRUCT: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_1_7B: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-1.7B",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_1_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOLLM2_360M

    sample_text = "The capital of France is"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SmolLM2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SmolLM2 model instance for this instance's variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the SmolLM2 model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits.argmax(dim=-1)
        predicted_text = self.tokenizer.decode(
            predicted_token_ids[0], skip_special_tokens=True
        )
        return predicted_text
