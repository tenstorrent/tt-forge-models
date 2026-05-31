# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-2 Expansion model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
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
    """Available GPT-2 Expansion model variants."""

    BASE = "gpt2_expansion"


class ModelLoader(ForgeModel):
    """polypo/gpt2-expansion loader for causal language modeling.

    polypo/gpt2-expansion is a standard GPT-2 architecture
    (GPT2LMHeadModel, 12 layers / 768 hidden / 12 heads, vocab 50257).
    """

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="polypo/gpt2-expansion",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "This is a sample text from "

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GPT-2-Expansion",
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
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        config = GPT2Config.from_pretrained(model_name)
        config_dict = config.to_dict()
        config_dict["use_cache"] = True
        if dtype_override is not None:
            config_dict["torch_dtype"] = dtype_override
        config = GPT2Config(**config_dict)

        model = GPT2LMHeadModel.from_pretrained(model_name, config=config, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Use random input for text generation, padded to a fixed length so the
        # last token is a (padded) zero - matches the gpt2 base loader pattern.
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

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
