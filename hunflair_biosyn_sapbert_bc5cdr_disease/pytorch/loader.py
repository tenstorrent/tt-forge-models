# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunFlair BioSyn SapBERT BC5CDR disease entity mention linker loader.
"""

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
    """Available HunFlair BioSyn SapBERT BC5CDR disease model variants."""

    BIOSYN_SAPBERT_BC5CDR_DISEASE = "biosyn-sapbert-bc5cdr-disease"


class ModelLoader(ForgeModel):
    """HunFlair BioSyn SapBERT BC5CDR disease entity mention linker loader."""

    _VARIANTS = {
        ModelVariant.BIOSYN_SAPBERT_BC5CDR_DISEASE: ModelConfig(
            pretrained_model_name="hunflair/biosyn-sapbert-bc5cdr-disease",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BIOSYN_SAPBERT_BC5CDR_DISEASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = (
            "The mutation in the ABCD1 gene causes X-linked adrenoleukodystrophy, "
            "a neurodegenerative disease."
        )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HunFlair_BioSyn_SapBERT_BC5CDR_Disease",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if torch.is_floating_point(v) else v
                for k, v in inputs.items()
            }
        return inputs

    def decode_output(self, outputs):
        embeddings = outputs.last_hidden_state[:, 0, :]
        print(f"Context: {self.sample_text}")
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings
