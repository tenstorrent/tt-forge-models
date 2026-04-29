# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bilingual-Embedding-Small model loader implementation for embedding generation.
"""
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
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
    """Available Bilingual-Embedding-Small model variants for embedding generation."""

    BILINGUAL_EMBEDDING_SMALL = "bilingual-embedding-small"


class ModelLoader(ForgeModel):
    """Bilingual-Embedding-Small model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.BILINGUAL_EMBEDDING_SMALL: ModelConfig(
            pretrained_model_name="Lajavaness/bilingual-embedding-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BILINGUAL_EMBEDDING_SMALL

    sample_sentences = ["This is an example sentence for embedding generation"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Bilingual-Embedding-Small",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # transformers 5.x removed is_decoder and add_cross_attention from
        # PretrainedConfig; BilingualConfig relies on inheriting those defaults.
        # Patch them onto the config before model __init__ reads them. Also set
        # return_dict on the config directly — passing return_dict=False to
        # from_pretrained with a pre-built config causes it to land in model
        # __init__ kwargs rather than being consumed as a config attribute.
        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        if not hasattr(config, "is_decoder"):
            config.is_decoder = False
        if not hasattr(config, "add_cross_attention"):
            config.add_cross_attention = False
        config.return_dict = False

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        # transformers 5.x uses init_empty_weights (meta device) during
        # from_pretrained; non-persistent buffers like token_type_ids are never
        # written by the checkpoint and come out uninitialized. Reset them.
        for module in model.modules():
            if hasattr(module, "token_type_ids"):
                module.token_type_ids.zero_()

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
