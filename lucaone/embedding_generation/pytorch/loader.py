# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LucaOne model loader implementation for embedding generation on
nucleic acid (DNA/RNA) and protein sequences.
"""

from transformers import AutoModel, AutoTokenizer
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
    """Available LucaOne model variants for embedding generation."""

    LUCAONE_DEFAULT_STEP36M = "LucaGroup/LucaOne-default-step36M"


class ModelLoader(ForgeModel):
    """LucaOne model loader implementation for embedding generation on
    nucleic acid (DNA/RNA) and protein sequences."""

    _VARIANTS = {
        ModelVariant.LUCAONE_DEFAULT_STEP36M: LLMModelConfig(
            pretrained_model_name="LucaGroup/LucaOne-default-step36M",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LUCAONE_DEFAULT_STEP36M

    # Sample nucleotide sequence for testing
    sample_sequence = "ATGCGTACGTTAGC"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LucaOne",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = dict(kwargs)
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        # LucaOne requires custom code and embedding-mode kwargs; these must
        # not be overridden by callers.
        model_kwargs["trust_remote_code"] = True
        model_kwargs["task_level"] = "token_level"
        model_kwargs["task_type"] = "embedding"

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # LucaOne's tokenizer requires a seq_type argument:
        # "gene" for DNA/RNA sequences, "prot" for protein sequences.
        inputs = self.tokenizer(
            self.sample_sequence,
            seq_type="gene",
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            token_embeddings = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state
        else:
            token_embeddings = outputs

        return token_embeddings
