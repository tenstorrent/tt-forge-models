# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-2 model loader implementations for text generation and sequence classification.
"""
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from typing import Optional, Union

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GPT-2 model variants."""

    GPT2_BASE = "gpt2"


class ModelLoader(ForgeModel):
    """GPT-2 loader for causal language modeling (text generation)."""

    _VARIANTS = {
        ModelVariant.GPT2_BASE: LLMModelConfig(
            pretrained_model_name="gpt2",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT2_BASE

    sample_text = "Once upon a time"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="gpt2",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.TEXT_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self):
        config = GPT2Config.from_pretrained(self._variant_config.pretrained_model_name)
        config_dict = config.to_dict()
        config_dict["return_dict"] = False
        config_dict["use_cache"] = False
        config = GPT2Config(**config_dict)
        model = GPT2LMHeadModel.from_pretrained(
            self._variant_config.pretrained_model_name, config=config
        )
        return model

    def load_inputs(self):
        if self.tokenizer is None:
            self._load_tokenizer()
        # Create input_ids: random tokens + zero padding as in original test
        input_ids = torch.cat(
            [
                torch.randint(
                    1,
                    GPT2Config.from_pretrained(
                        self._variant_config.pretrained_model_name
                    ).vocab_size,
                    (1, 255),
                ),
                torch.zeros(1, 1, dtype=torch.int64),
            ],
            dim=-1,
        ).to(torch.int64)
        return {"input_ids": input_ids}

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.
        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs
        Returns:
            str: Predicted sentiment label
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        # Get the logits from the outputs
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Get the predicted class ID
        predicted_value = logits.argmax(-1).item()

        # Get the model to access config
        model = self.load_model()
        predicted_sentiment = model.config.id2label[predicted_value]

        return predicted_sentiment
