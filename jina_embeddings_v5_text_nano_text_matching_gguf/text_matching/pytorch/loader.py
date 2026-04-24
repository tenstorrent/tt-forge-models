# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina Embeddings v5 Text Nano Text Matching GGUF model loader implementation for text matching.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional

# Source of modeling_eurobert.py and config.json with auto_map; the GGUF repo
# has no config.json or model code so we pull class definitions from here.
_EUROBERT_BASE_MODEL = "jinaai/jina-embeddings-v5-text-nano-text-matching"


def _patch_transformers_eurobert_gguf():
    """Monkey-patch transformers to add eurobert GGUF architecture support.

    Transformers lacks GGUF loading support for the eurobert architecture.
    EuroBERT inherits from LlamaConfig/LlamaModel, so we reuse the llama
    config key mapping and tensor processor.  We also register the EuroBERT
    config and model classes with the Auto factories so they are resolved by
    model_type without requiring an auto_map entry in the checkpoint.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        LlamaTensorProcessor,
        TENSOR_PROCESSORS,
    )

    if "eurobert" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # 1. Add config key mapping for eurobert (llama-style encoder)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["eurobert"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "vocab_size": "vocab_size",
    }

    # 2. Register eurobert as a supported GGUF architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("eurobert")

    # 3. Register tokenizer converter (eurobert uses GPT-2-style BPE tokenizer)
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    if "eurobert" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["eurobert"] = GGUFGPTConverter

    # 4. Register tensor processor (eurobert uses same Q/K weight layout as llama)
    if "eurobert" not in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["eurobert"] = LlamaTensorProcessor

    # 5. Register EuroBERT config and model classes with the Auto factories.
    # The GGUF repo has no config.json so there's no auto_map; downloading the
    # class definitions from the base model and registering them lets AutoConfig
    # and AutoModel resolve "eurobert" model_type without trust_remote_code lookups
    # against the GGUF repo.
    from transformers import AutoConfig, AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_class = get_class_from_dynamic_module(
        "configuration_eurobert.EuroBertConfig",
        _EUROBERT_BASE_MODEL,
        trust_remote_code=True,
    )
    model_class = get_class_from_dynamic_module(
        "modeling_eurobert.EuroBertModel",
        _EUROBERT_BASE_MODEL,
        trust_remote_code=True,
    )
    AutoConfig.register("eurobert", config_class, exist_ok=True)
    AutoModel.register(config_class, model_class, exist_ok=True)


_patch_transformers_eurobert_gguf()

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
    """Available Jina Embeddings v5 Text Nano Text Matching GGUF model variants."""

    JINA_EMBEDDINGS_V5_TEXT_NANO_TEXT_MATCHING_GGUF = (
        "jina-embeddings-v5-text-nano-text-matching-GGUF"
    )


class ModelLoader(ForgeModel):
    """Jina Embeddings v5 Text Nano Text Matching GGUF model loader for text matching."""

    _VARIANTS = {
        ModelVariant.JINA_EMBEDDINGS_V5_TEXT_NANO_TEXT_MATCHING_GGUF: ModelConfig(
            pretrained_model_name="jinaai/jina-embeddings-v5-text-nano-text-matching-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JINA_EMBEDDINGS_V5_TEXT_NANO_TEXT_MATCHING_GGUF

    GGUF_FILE = "v5-nano-text-matching-Q4_K_M.gguf"

    sample_sentences = [
        "Jina Embeddings v5 is a multilingual text embedding model for text matching"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Jina-Embeddings-v5-Text-Nano-Text-Matching-GGUF",
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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        # Last-token pooling: select the last non-padded token for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(token_embeddings.size(0))
        sentence_embeddings = token_embeddings[batch_indices, seq_lengths]

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
