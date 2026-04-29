# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE-M3 GGUF model loader implementation for embedding generation.
"""
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional

# transformers 5.x does not include "bert" in its GGUF model-loading
# machinery.  Register it so that AutoModel.from_pretrained(gguf_file=…)
# can load BERT-architecture GGUF checkpoints (e.g. gpustack/bge-m3-GGUF).
#
# Tables patched:
#   GGUF_CONFIG_MAPPING        – maps GGUF field names → BertConfig fields
#   GGUF_SUPPORTED_ARCHITECTURES – architecture allow-list checked at load
#   GGUF_CONFIG_DEFAULTS_MAPPING – type_vocab_size=1 so the single
#                                  token-type embedding row matches the GGUF
#   GGUF_TO_FAST_CONVERTERS    – SentencePiece converter for future callers
#                                  (this loader bypasses GGUF tokenizer
#                                  loading; see TOKENIZER_MODEL_NAME below)
#
# A custom TensorProcessor reshapes the 1-D token_types.weight stored in
# the GGUF file ([hidden_size]) to the 2-D shape ([1, hidden_size]) that
# BertModel's Embedding layer expects when type_vocab_size=1.
from transformers.integrations.ggml import (
    GGUF_CONFIG_MAPPING,
    GGUF_CONFIG_DEFAULTS_MAPPING,
    GGUF_TO_FAST_CONVERTERS,
    GGUFT5Converter,
)
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import TensorProcessor, GGUFTensor


class _BertTensorProcessor(TensorProcessor):
    def process(self, weights, name, **kwargs):
        if name == "token_types.weight" and weights.ndim == 1:
            weights = weights.reshape(1, -1)
        return GGUFTensor(weights, name, {})


if "bert" not in GGUF_CONFIG_MAPPING:
    GGUF_CONFIG_MAPPING["bert"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
    }
    GGUF_CONFIG_DEFAULTS_MAPPING["bert"] = {"type_vocab_size": 1}
    GGUF_TO_FAST_CONVERTERS["bert"] = GGUFT5Converter
    _gguf_utils.TENSOR_PROCESSORS["bert"] = _BertTensorProcessor
    if "bert" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("bert")

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
    """Available BGE-M3 GGUF model variants."""

    BGE_M3_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """BGE-M3 GGUF model loader implementation for embedding generation tasks."""

    _VARIANTS = {
        ModelVariant.BGE_M3_Q4_K_M: ModelConfig(
            pretrained_model_name="gpustack/bge-m3-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BGE_M3_Q4_K_M

    GGUF_FILE = "bge-m3-Q4_K_M.gguf"

    sample_texts = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BGE-M3 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # The GGUF repo only ships the quantized weights file; the tokenizer
    # files (tokenizer.json, sentencepiece.bpe.model, etc.) live on the
    # original BAAI/bge-m3 hub page.  We load them from there so that
    # token IDs stay inside the 250002-token vocabulary and we get the
    # correct XLMRobertaTokenizer (not BertTokenizerFast, which appends
    # extra special-token IDs above vocab_size).
    TOKENIZER_MODEL_NAME = "BAAI/bge-m3"

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_MODEL_NAME)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
