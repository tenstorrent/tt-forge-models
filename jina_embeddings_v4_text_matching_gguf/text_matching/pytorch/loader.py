# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina Embeddings v4 Text Matching GGUF model loader implementation for text matching.
"""
import torch
from transformers import AutoModel, AutoTokenizer
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


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    jina-embeddings-v4 GGUF files use the 'qwen2vl' architecture identifier
    but contain only language-model tensors (no vision encoder). Transformers
    5.2.x does not list qwen2vl in GGUF_SUPPORTED_ARCHITECTURES. We bridge
    the gap by registering the config mapping and remapping model_type to
    qwen2 after loading.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
    )

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # Register qwen2vl as a supported architecture.
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    # Map qwen2vl GGUF config fields to Qwen2 HF config fields.
    # qwen2vl uses the same LM config structure as qwen2; the only difference
    # is rope.dimension_sections (M-RoPE) instead of rope.dimension_count.
    # We ignore rope.dimension_sections since the loaded Qwen2Model uses
    # standard rotary embeddings derived from head_dim automatically.
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "rope.dimension_sections": None,  # discard; not used by Qwen2Model
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    # Use the existing qwen2 fast tokenizer converter for qwen2vl.
    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS and "qwen2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUF_TO_FAST_CONVERTERS["qwen2"]

    # Wrap load_gguf_checkpoint to remap model_type from qwen2vl to qwen2.
    # Several modules import load_gguf_checkpoint by value (not by module
    # reference), so we must patch each already-bound name individually.
    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _tok_auto

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            config["model_type"] = "qwen2"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    _tok_auto.load_gguf_checkpoint = patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Jina Embeddings v4 Text Matching GGUF model variants."""

    JINA_EMBEDDINGS_V4_TEXT_MATCHING_GGUF = "jina-embeddings-v4-text-matching-GGUF"


class ModelLoader(ForgeModel):
    """Jina Embeddings v4 Text Matching GGUF model loader for text matching."""

    _VARIANTS = {
        ModelVariant.JINA_EMBEDDINGS_V4_TEXT_MATCHING_GGUF: ModelConfig(
            pretrained_model_name="jinaai/jina-embeddings-v4-text-matching-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JINA_EMBEDDINGS_V4_TEXT_MATCHING_GGUF

    GGUF_FILE = "jina-embeddings-v4-text-matching-Q4_K_M.gguf"

    sample_sentences = ["Query: Jina Embeddings v4 is a text matching embedding model"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Jina-Embeddings-v4-Text-Matching-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_transformers_qwen2vl_gguf()

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _patch_transformers_qwen2vl_gguf()

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

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
