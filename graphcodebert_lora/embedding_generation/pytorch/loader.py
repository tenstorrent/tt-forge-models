# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GraphCodeBERT LoRA model loader implementation for embedding generation.

This model applies a LoRA adapter (Shreshth13/graphcodebert-lora) on top of the
microsoft/graphcodebert-base model for code embedding/feature extraction.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GraphCodeBERT LoRA model variants."""

    GRAPHCODEBERT_LORA = "GraphCodeBERT_LoRA"


class ModelLoader(ForgeModel):
    """GraphCodeBERT LoRA model loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.GRAPHCODEBERT_LORA: ModelConfig(
            pretrained_model_name="Shreshth13/graphcodebert-lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRAPHCODEBERT_LORA

    BASE_MODEL_NAME = "microsoft/graphcodebert-base"

    sample_code = "def hello_world():\n    print('Hello, World!')"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GraphCodeBERT-LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GraphCodeBERT base model with LoRA adapter applied.

        Returns:
            torch.nn.Module: The merged LoRA model instance.
        """
        from transformers import AutoModel, AutoTokenizer
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_NAME)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = AutoModel.from_pretrained(self.BASE_MODEL_NAME, **model_kwargs)
        model = PeftModel.from_pretrained(
            base_model, self._variant_config.pretrained_model_name
        )
        model = model.merge_and_unload()
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample inputs for the model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_NAME)

        inputs = self.tokenizer(
            self.sample_code,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
