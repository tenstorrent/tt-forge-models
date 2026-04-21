# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nomic Embed Text v1 GGUF model loader implementation for sentence embedding generation.
"""
import re

import torch
from huggingface_hub import hf_hub_download
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

_GGUF_TO_HF_NAME = {
    "token_embd.weight": "embeddings.word_embeddings.weight",
    "token_types.weight": "embeddings.token_type_embeddings.weight",
    "token_embd_norm.weight": "emb_ln.weight",
    "token_embd_norm.bias": "emb_ln.bias",
}

_BLK_SUFFIX_MAP = {
    "attn_qkv.weight": "attn.Wqkv.weight",
    "attn_output.weight": "attn.out_proj.weight",
    "ffn_up.weight": "mlp.fc11.weight",
    "ffn_gate.weight": "mlp.fc12.weight",
    "ffn_down.weight": "mlp.fc2.weight",
    "attn_output_norm.weight": "norm1.weight",
    "attn_output_norm.bias": "norm1.bias",
    "layer_output_norm.weight": "norm2.weight",
    "layer_output_norm.bias": "norm2.bias",
}


def _gguf_name_to_hf(name):
    if name in _GGUF_TO_HF_NAME:
        return _GGUF_TO_HF_NAME[name]
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if m and m.group(2) in _BLK_SUFFIX_MAP:
        return f"encoder.layers.{m.group(1)}.{_BLK_SUFFIX_MAP[m.group(2)]}"
    return None


def _load_model_from_gguf(pretrained_model_name, gguf_file, config_repo, model_kwargs):
    """Load a nomic-bert model by dequantizing a GGUF file into a freshly built model."""
    from gguf import GGUFReader, dequantize

    gguf_path = hf_hub_download(repo_id=pretrained_model_name, filename=gguf_file)

    config = AutoConfig.from_pretrained(config_repo, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    model_sd = model.state_dict()

    reader = GGUFReader(gguf_path)
    loaded_sd = {}
    for tensor in reader.tensors:
        hf_name = _gguf_name_to_hf(tensor.name)
        if hf_name is None:
            continue
        weights = dequantize(tensor.data, tensor.tensor_type)
        # dequantize returns tensors already in PyTorch (row-major, [out, in]) convention.
        pt = torch.from_numpy(weights.copy())
        # Pad vocabulary dimension if the model uses a rounded vocab size.
        if hf_name == "embeddings.word_embeddings.weight":
            target_vocab = model_sd[hf_name].shape[0]
            if pt.shape[0] < target_vocab:
                pad = torch.zeros(
                    target_vocab - pt.shape[0],
                    pt.shape[1],
                    dtype=pt.dtype,
                )
                pt = torch.cat([pt, pad], dim=0)
        loaded_sd[hf_name] = pt.to(model_sd[hf_name].dtype)

    model.load_state_dict(loaded_sd, strict=False)
    model.eval()
    return model


class ModelVariant(StrEnum):
    """Available Nomic Embed Text v1 GGUF model variants for embedding generation."""

    NOMIC_EMBED_TEXT_V1_GGUF = "nomic-embed-text-v1-GGUF"


class ModelLoader(ForgeModel):
    """Nomic Embed Text v1 GGUF model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.NOMIC_EMBED_TEXT_V1_GGUF: ModelConfig(
            pretrained_model_name="nomic-ai/nomic-embed-text-v1-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOMIC_EMBED_TEXT_V1_GGUF

    GGUF_FILE = "nomic-embed-text-v1.Q4_K_M.gguf"

    # Non-GGUF repo used to obtain the model config and custom architecture code.
    CONFIG_REPO = "nomic-ai/nomic-embed-text-v1"

    sample_sentences = [
        "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nomic-Embed-Text-v1-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.CONFIG_REPO)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model = _load_model_from_gguf(
            self._variant_config.pretrained_model_name,
            self.GGUF_FILE,
            self.CONFIG_REPO,
            kwargs,
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
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
