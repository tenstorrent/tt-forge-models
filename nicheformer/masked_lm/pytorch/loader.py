# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nicheformer model loader implementation for masked language modeling on single-cell data.
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
    """Available Nicheformer model variants for masked language modeling."""

    NICHEFORMER = "nicheformer"


class ModelLoader(ForgeModel):
    """Nicheformer model loader implementation for masked language modeling.

    Nicheformer is a transformer foundation model for single-cell and spatial
    omics data. The published tokenizer consumes AnnData objects and a
    `technology_mean.npy` file, which are not suitable for compile-only tests;
    synthetic token IDs are used instead.
    """

    _VARIANTS = {
        ModelVariant.NICHEFORMER: ModelConfig(
            pretrained_model_name="theislab/Nicheformer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NICHEFORMER

    # Matches the `n_tokens` and `context_length` fields of the model config.
    vocab_size = 20340
    seq_length = 1500

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Nicheformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForMaskedLM

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        input_ids = torch.randint(1, self.vocab_size, (1, self.seq_length))
        attention_mask = torch.ones(1, self.seq_length, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
