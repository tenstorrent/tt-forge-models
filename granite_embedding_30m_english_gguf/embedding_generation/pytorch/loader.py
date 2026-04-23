# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite Embedding 30M English GGUF model loader implementation for embedding generation.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_CONFIG_MAPPING, GGUF_TO_FAST_CONVERTERS

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

_BERT_GGUF_CONFIG_MAPPING = {
    "block_count": "num_hidden_layers",
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "attention.head_count": "num_attention_heads",
    "attention.layer_norm_epsilon": "layer_norm_eps",
}


def _patch_bert_gguf_support():
    """Register bert as a supported GGUF architecture.

    Adds config key mappings and tokenizer converter for BERT GGUF models.
    The tokenizer uses a GPT2/BPE format (roberta-bpe pre-tokenizer), so we
    reuse the gpt2 fast converter.
    """
    if "bert" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["bert"] = dict(_BERT_GGUF_CONFIG_MAPPING)
    if "bert" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("bert")
    if "bert" not in GGUF_TO_FAST_CONVERTERS and "gpt2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["bert"] = GGUF_TO_FAST_CONVERTERS["gpt2"]


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    _patch_bert_gguf_support()
    return _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )


_patch_bert_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Granite Embedding 30M English GGUF model variants for embedding generation."""

    GRANITE_EMBEDDING_30M_ENGLISH_GGUF = "granite-embedding-30m-english-GGUF"


class ModelLoader(ForgeModel):
    """Granite Embedding 30M English GGUF model loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.GRANITE_EMBEDDING_30M_ENGLISH_GGUF: ModelConfig(
            pretrained_model_name="bartowski/granite-embedding-30m-english-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_EMBEDDING_30M_ENGLISH_GGUF

    GGUF_FILE = "granite-embedding-30m-english-Q4_K_M.gguf"

    # The base (non-GGUF) model for tokenizer loading; the GGUF converter
    # produces out-of-range token IDs for this RoBERTa-style tokenizer.
    TOKENIZER_MODEL = "ibm-granite/granite-embedding-30m-english"

    sample_sentences = [
        "Who made the song My achy breaky heart?",
        "summit define",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Granite-Embedding-30M-English-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # Load from the base model: the GGUF tokenizer converter produces
        # out-of-range token IDs for this RoBERTa-style tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_MODEL)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
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

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        # CLS pooling followed by L2 normalization
        sentence_embeddings = token_embeddings[:, 0]
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
