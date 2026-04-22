# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SLUAR model loader implementation for authorship embedding generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, BertConfig
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


class LUARModel(nn.Module):
    """LUAR/SLUAR model: MiniLM transformer + linear projection for authorship embeddings."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        bert_config = BertConfig(
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=1536,
            max_position_embeddings=512,
            vocab_size=30522,
            type_vocab_size=2,
        )
        self.transformer = BertModel(bert_config)
        self.linear = nn.Linear(384, embedding_dim)

    def _mean_pool(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(token_embeddings.dtype)
        )
        return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        projected = self.linear(pooled)
        return F.normalize(projected, p=2, dim=-1)


class ModelVariant(StrEnum):
    """Available SLUAR model variants."""

    SLUAR = "sluar"


class ModelLoader(ForgeModel):
    """SLUAR model loader implementation for authorship embedding generation."""

    _VARIANTS = {
        ModelVariant.SLUAR: ModelConfig(
            pretrained_model_name="noandrews/sluar",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SLUAR

    sample_sentences = [
        "This is an example sentence for authorship embedding generation."
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="sluar",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # The sluar checkpoint has no tokenizer files; use the MiniLM backbone tokenizer directly.
    _TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from safetensors.torch import load_file
        import huggingface_hub

        pretrained_model_name = self._variant_config.pretrained_model_name
        weights_path = huggingface_hub.hf_hub_download(
            pretrained_model_name, "model.safetensors"
        )

        model = LUARModel(embedding_dim=512)
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
