# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
all-MiniLM-L12-v2 GGUF model loader implementation for embedding generation.
"""
import re
from typing import Optional

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available all-MiniLM-L12-v2 GGUF model variants."""

    ALL_MINILM_L12_V2_Q4_K_M = "Q4_K_M"


_GGUF_TO_HF_EMBED = {
    "token_embd.weight": "embeddings.word_embeddings.weight",
    "position_embd.weight": "embeddings.position_embeddings.weight",
    "token_types.weight": "embeddings.token_type_embeddings.weight",
    "token_embd_norm.weight": "embeddings.LayerNorm.weight",
    "token_embd_norm.bias": "embeddings.LayerNorm.bias",
}

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_BLK_SUBMAP = {
    "attn_q.weight": "attention.self.query.weight",
    "attn_q.bias": "attention.self.query.bias",
    "attn_k.weight": "attention.self.key.weight",
    "attn_k.bias": "attention.self.key.bias",
    "attn_v.weight": "attention.self.value.weight",
    "attn_v.bias": "attention.self.value.bias",
    "attn_output.weight": "attention.output.dense.weight",
    "attn_output.bias": "attention.output.dense.bias",
    "attn_output_norm.weight": "attention.output.LayerNorm.weight",
    "attn_output_norm.bias": "attention.output.LayerNorm.bias",
    "ffn_up.weight": "intermediate.dense.weight",
    "ffn_up.bias": "intermediate.dense.bias",
    "ffn_down.weight": "output.dense.weight",
    "ffn_down.bias": "output.dense.bias",
    "layer_output_norm.weight": "output.LayerNorm.weight",
    "layer_output_norm.bias": "output.LayerNorm.bias",
}


def _gguf_name_to_hf(name: str) -> Optional[str]:
    if name in _GGUF_TO_HF_EMBED:
        return _GGUF_TO_HF_EMBED[name]
    m = _BLK_RE.match(name)
    if m:
        layer_idx, sub = m.group(1), m.group(2)
        if sub in _BLK_SUBMAP:
            return f"encoder.layer.{layer_idx}.{_BLK_SUBMAP[sub]}"
    return None


def _load_bert_from_gguf(
    gguf_path: str, dtype: torch.dtype = torch.float32
) -> BertModel:
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(gguf_path)
    fields = {f.name: f for f in reader.fields.values()}

    def _field_int(key: str) -> int:
        return int(fields[key].parts[-1][0])

    config = BertConfig(
        hidden_size=_field_int("bert.embedding_length"),
        num_hidden_layers=_field_int("bert.block_count"),
        num_attention_heads=_field_int("bert.attention.head_count"),
        intermediate_size=_field_int("bert.feed_forward_length"),
        max_position_embeddings=_field_int("bert.context_length"),
        vocab_size=30522,
        type_vocab_size=2,
        hidden_act="gelu",
    )

    model = BertModel(config, add_pooling_layer=False)

    state_dict = {}
    for tensor in reader.tensors:
        hf_name = _gguf_name_to_hf(tensor.name)
        if hf_name is None:
            continue
        data = dequantize(tensor.data, tensor.tensor_type)
        state_dict[hf_name] = torch.tensor(data, dtype=dtype)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_buffer_missing = [k for k in missing if "position_ids" not in k]
    if non_buffer_missing:
        raise ValueError(f"Missing BERT weights from GGUF: {non_buffer_missing}")

    return model.eval()


class ModelLoader(ForgeModel):
    """all-MiniLM-L12-v2 GGUF model loader implementation for embedding generation tasks."""

    _VARIANTS = {
        ModelVariant.ALL_MINILM_L12_V2_Q4_K_M: ModelConfig(
            pretrained_model_name="leliuga/all-MiniLM-L12-v2-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALL_MINILM_L12_V2_Q4_K_M

    GGUF_FILE = "all-MiniLM-L12-v2.Q4_K_M.gguf"
    TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

    sample_texts = [
        "This is an example sentence for embedding generation.",
        "Each sentence is converted into a fixed-length vector.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="all-MiniLM-L12-v2 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_MODEL)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = hf_hub_download(
            self._variant_config.pretrained_model_name, self.GGUF_FILE
        )
        dtype = dtype_override if dtype_override is not None else torch.float32
        model = _load_bert_from_gguf(gguf_path, dtype=dtype)
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
            max_length=256,
        )
        return inputs

    def load_config(self):
        self.load_model()
        return self.config
