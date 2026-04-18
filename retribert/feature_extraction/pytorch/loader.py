# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RetriBERT model loader for text retrieval feature extraction.

RetriBERT was removed from transformers v5, so this loader manually
reconstructs the model from BertModel + projection layers.
"""
import torch
import torch.nn as nn
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class RetriBertModel(nn.Module):
    """RetriBERT: BERT encoder with query/doc projection heads.

    Reimplemented here because transformers v5 removed RetriBertModel.
    Architecture: shared BertModel (8 layers) + two linear projections.
    """

    def __init__(self, bert, project_query, project_doc):
        super().__init__()
        self.bert_query = bert
        self.project_query = project_query
        self.project_doc = project_doc

    def embed_questions(self, input_ids, attention_mask=None):
        token_type_ids = torch.zeros_like(input_ids)
        cls_output = self.bert_query(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[0][:, 0, :]
        return self.project_query(cls_output)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.embed_questions(input_ids, attention_mask=attention_mask)


class ModelVariant(StrEnum):
    """Available model variants for RetriBERT."""

    RETRIBERT_BASE_UNCASED = "yjernite/retribert-base-uncased"


class ModelLoader(ForgeModel):
    """RetriBERT model loader for text retrieval feature extraction."""

    _VARIANTS = {
        ModelVariant.RETRIBERT_BASE_UNCASED: LLMModelConfig(
            pretrained_model_name="yjernite/retribert-base-uncased",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RETRIBERT_BASE_UNCASED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RetriBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        return self.tokenizer

    @staticmethod
    def _build_retribert(model_name, dtype_override=None):
        from transformers import BertConfig, BertModel
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_name, "config.json")
        import json

        with open(config_path) as f:
            raw_config = json.load(f)

        bert_config = BertConfig(
            vocab_size=raw_config["vocab_size"],
            hidden_size=raw_config["hidden_size"],
            num_hidden_layers=raw_config["num_hidden_layers"],
            num_attention_heads=raw_config["num_attention_heads"],
            intermediate_size=raw_config["intermediate_size"],
            hidden_act=raw_config["hidden_act"],
            hidden_dropout_prob=raw_config["hidden_dropout_prob"],
            attention_probs_dropout_prob=raw_config["attention_probs_dropout_prob"],
            max_position_embeddings=raw_config["max_position_embeddings"],
            type_vocab_size=raw_config["type_vocab_size"],
            layer_norm_eps=raw_config["layer_norm_eps"],
            pad_token_id=raw_config["pad_token_id"],
        )
        if dtype_override is not None:
            bert_config.torch_dtype = dtype_override

        hidden_size = raw_config["hidden_size"]
        projection_dim = raw_config["projection_dim"]

        bert = BertModel(bert_config)
        project_query = nn.Linear(hidden_size, projection_dim, bias=False)
        project_doc = nn.Linear(hidden_size, projection_dim, bias=False)

        weights_path = hf_hub_download(model_name, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        bert_sd = {
            k.replace("bert_query.", ""): v
            for k, v in state_dict.items()
            if k.startswith("bert_query.")
        }
        bert.load_state_dict(bert_sd)
        project_query.weight = nn.Parameter(state_dict["project_query.weight"])
        project_doc.weight = nn.Parameter(state_dict["project_doc.weight"])

        model = RetriBertModel(bert, project_query, project_doc)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name
        model = self._build_retribert(model_name, dtype_override=dtype_override)
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if sentence is None:
            sentence = "How many people live in Berlin?"

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, (tuple, list)):
            return fwd_output[0].flatten()
        return fwd_output.flatten()
