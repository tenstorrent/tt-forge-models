# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-2 model loader implementations for text generation and sequence classification.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)
from typing import Optional

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

    GPT2_BASE = "Default"
    GPT2_BS3V2_XSUM = "bs3v2-xsum"
    GPT2_LARGE = "Large"
    GPT2_LUMELETO = "Lumeleto"
    GPT2_SEQUENCE_CLASSIFICATION = "Sequence_Classification"
    GPT2_COLA = "Cola"
    TINY_RANDOM = "tiny-random"
    TINY_RANDOM_GEN_CONFIG = "tiny-random-gen-config"


class ModelLoader(ForgeModel):
    """GPT-2 loader for causal language modeling and sequence classification."""

    _VARIANTS = {
        ModelVariant.GPT2_BASE: LLMModelConfig(
            pretrained_model_name="gpt2",
            max_length=256,
        ),
        ModelVariant.GPT2_BS3V2_XSUM: LLMModelConfig(
            pretrained_model_name="nbtpj/bs3v2_gpt2_xsum",
            max_length=256,
        ),
        ModelVariant.GPT2_LARGE: LLMModelConfig(
            pretrained_model_name="openai-community/gpt2-large",
            max_length=256,
        ),
        ModelVariant.GPT2_LUMELETO: LLMModelConfig(
            pretrained_model_name="gratefulasi/lumeleto",
            max_length=256,
        ),
        ModelVariant.GPT2_SEQUENCE_CLASSIFICATION: LLMModelConfig(
            pretrained_model_name="mnoukhov/gpt2-imdb-sentiment-classifier",
            max_length=256,
        ),
        ModelVariant.GPT2_COLA: LLMModelConfig(
            pretrained_model_name="tanganke/gpt2_cola",
            max_length=256,
        ),
        ModelVariant.TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="peft-internal-testing/tiny-random-gpt2",
            max_length=256,
        ),
        ModelVariant.TINY_RANDOM_GEN_CONFIG: LLMModelConfig(
            pretrained_model_name="joaogante/tiny-random-gpt2-with-generation-config",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT2_BASE

    sample_text = "This is a sample text from "

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

        if variant in (
            ModelVariant.GPT2_SEQUENCE_CLASSIFICATION,
            ModelVariant.GPT2_COLA,
        ):
            task = ModelTask.NLP_TEXT_CLS
        else:
            task = ModelTask.NLP_CAUSAL_LM

        group = (
            ModelGroup.VULCAN
            if variant
            in (
                ModelVariant.GPT2_LARGE,
                ModelVariant.TINY_RANDOM,
                ModelVariant.GPT2_COLA,
            )
            else ModelGroup.GENERALITY
        )

        return ModelInfo(
            model="GPT-2",
            variant=variant,
            group=group,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        # Set padding side to left for classification variants
        if self._variant in (
            ModelVariant.GPT2_SEQUENCE_CLASSIFICATION,
            ModelVariant.GPT2_COLA,
        ):
            self.tokenizer.padding_side = "left"

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        if self._variant in (
            ModelVariant.GPT2_BASE,
            ModelVariant.GPT2_BS3V2_XSUM,
            ModelVariant.GPT2_LARGE,
            ModelVariant.TINY_RANDOM,
            ModelVariant.TINY_RANDOM_GEN_CONFIG,
        ):
            config = GPT2Config.from_pretrained(model_name)
            config_dict = config.to_dict()
            config_dict["use_cache"] = True
            if dtype_override is not None:
                config_dict["torch_dtype"] = dtype_override
            if self.num_layers is not None:
                config_dict["num_hidden_layers"] = self.num_layers
            config = GPT2Config(**config_dict)
            model = GPT2LMHeadModel.from_pretrained(model_name, config=config, **kwargs)
        else:
            model_kwargs = {
                "trust_remote_code": True,
                "use_cache": False,
            }
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **model_kwargs
            )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if self._variant in (
            ModelVariant.GPT2_BASE,
            ModelVariant.GPT2_BS3V2_XSUM,
            ModelVariant.GPT2_LARGE,
            ModelVariant.TINY_RANDOM,
            ModelVariant.TINY_RANDOM_GEN_CONFIG,
        ):
            # Use random input for text generation
            vocab_size = GPT2Config.from_pretrained(
                self._variant_config.pretrained_model_name
            ).vocab_size

            input_ids = torch.cat(
                [
                    torch.randint(1, vocab_size, (1, 255)),
                    torch.zeros(1, 1, dtype=torch.int64),
                ],
                dim=-1,
            ).to(torch.int64)

            return {"input_ids": input_ids}

        elif self._variant == ModelVariant.GPT2_MNLI:
            premise = "The new rights are nice enough."
            hypothesis = "Everyone really likes the newest benefits."
            tokenized = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._variant_config.max_length,
            )
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }

        else:
            test_input = self.sample_text
            tokenized = self.tokenizer(
                test_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._variant_config.max_length,
            )
            return {"input_ids": tokenized["input_ids"]}

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        if self._variant in (
            ModelVariant.GPT2_SEQUENCE_CLASSIFICATION,
            ModelVariant.GPT2_COLA,
        ):
            # For classification: map class index to label
            predicted_value = logits.argmax(-1).item()
            model = self.load_model()
            return model.config.id2label[predicted_value]
        else:
            # For generation: decode tokens
            generated_ids = logits.argmax(-1)
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
