# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE-M3 GGUF model loader implementation for embedding generation.
"""
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
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


def _patch_transformers_bert_gguf():
    """Monkey-patch transformers to add BERT GGUF architecture support.

    The gguf library already knows about BERT tensor names, but
    transformers lacks the config mapping and architecture registration
    needed to load BERT GGUF checkpoints.  BGE-M3 is an XLM-RoBERTa
    model stored in GGUF with architecture="bert".
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "bert" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register bert as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("bert")

    # 2. Add config mapping for bert -> XLMRobertaConfig fields
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["bert"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
        "attention.causal": None,
        "pooling_type": None,
    }

    # 3. Register SentencePiece tokenizer converter for bert (uses t5/SPM format)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFT5Converter,
    )

    for name in ("bert", "xlm-roberta"):
        if name not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS[name] = GGUFT5Converter

    # 4. Patch load_gguf_checkpoint to set correct model_type and architectures
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "bert":
            config["model_type"] = "xlm-roberta"
            config["architectures"] = ["XLMRobertaModel"]
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Also patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


# Apply the monkey-patch at import time
_patch_transformers_bert_gguf()


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

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

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
