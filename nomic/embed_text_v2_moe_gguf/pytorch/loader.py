# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nomic Embed Text v2 MoE GGUF model loader implementation for sentence embedding generation.

Transformers does not natively support loading GGUF files for the nomic-bert-moe
architecture.  This loader works around the gap by:
  1. Instantiating the model architecture from the non-GGUF source repo
     (nomic-ai/nomic-embed-text-v2-moe) via ``trust_remote_code``.
  2. Downloading the quantized GGUF file from the GGUF repo
     (nomic-ai/nomic-embed-text-v2-moe-GGUF).
  3. Dequantizing and mapping each GGUF tensor to the corresponding HF
     parameter name, then loading the state dict into the model.
"""
import numpy as np
import torch
from gguf import GGUFReader, dequantize
from huggingface_hub import hf_hub_download
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

# Source repo with the model architecture and tokenizer (trust_remote_code)
_SOURCE_REPO = "nomic-ai/nomic-embed-text-v2-moe"


def _build_gguf_to_hf_map(num_blocks: int):
    """Build a mapping from GGUF tensor names to HF parameter names."""
    m = {
        "token_embd.weight": "embeddings.word_embeddings.weight",
        "token_types.weight": "embeddings.token_type_embeddings.weight",
        "token_embd_norm.weight": "emb_ln.weight",
        "token_embd_norm.bias": "emb_ln.bias",
    }
    for n in range(num_blocks):
        prefix_g = f"blk.{n}"
        prefix_h = f"encoder.layers.{n}"
        # Attention (shared by all blocks)
        m[f"{prefix_g}.attn_qkv.weight"] = f"{prefix_h}.attn.Wqkv.weight"
        m[f"{prefix_g}.attn_qkv.bias"] = f"{prefix_h}.attn.Wqkv.bias"
        m[f"{prefix_g}.attn_output.weight"] = f"{prefix_h}.attn.out_proj.weight"
        m[f"{prefix_g}.attn_output.bias"] = f"{prefix_h}.attn.out_proj.bias"
        # Layer norms
        m[f"{prefix_g}.attn_output_norm.weight"] = f"{prefix_h}.norm1.weight"
        m[f"{prefix_g}.attn_output_norm.bias"] = f"{prefix_h}.norm1.bias"
        m[f"{prefix_g}.layer_output_norm.weight"] = f"{prefix_h}.norm2.weight"
        m[f"{prefix_g}.layer_output_norm.bias"] = f"{prefix_h}.norm2.bias"
        # Dense FFN (even blocks)
        m[f"{prefix_g}.ffn_up.weight"] = f"{prefix_h}.mlp.fc1.weight"
        m[f"{prefix_g}.ffn_up.bias"] = f"{prefix_h}.mlp.fc1.bias"
        m[f"{prefix_g}.ffn_down.weight"] = f"{prefix_h}.mlp.fc2.weight"
        m[f"{prefix_g}.ffn_down.bias"] = f"{prefix_h}.mlp.fc2.bias"
        # MoE (odd blocks)
        m[f"{prefix_g}.ffn_gate_inp.weight"] = f"{prefix_h}.mlp.router.layer.weight"
        m[f"{prefix_g}.ffn_up_exps.weight"] = f"{prefix_h}.mlp.experts.mlp.w1"
        m[f"{prefix_g}.ffn_down_exps.weight"] = f"{prefix_h}.mlp.experts.mlp.w2"
    return m


def _load_gguf_into_model(model, gguf_path, dtype):
    """Dequantize GGUF tensors and load them into *model* in-place."""
    reader = GGUFReader(gguf_path)
    num_blocks = model.config.n_layer
    name_map = _build_gguf_to_hf_map(num_blocks)
    state_dict = model.state_dict()

    for tensor in reader.tensors:
        hf_name = name_map.get(tensor.name)
        if hf_name is None:
            continue

        weights = dequantize(tensor.data, tensor.tensor_type)

        # Restore the original GGUF shape (dequantize may flatten dimensions)
        orig_shape = tuple(int(d) for d in tensor.shape)
        if weights.shape != orig_shape:
            weights = weights.reshape(orig_shape)

        # MoE expert weights: GGUF stores as 3-D, HF packs as 2-D.
        # Both w1 and w2 in HF are [num_experts * ffn_dim, hidden_dim].
        if "ffn_up_exps" in tensor.name:
            # GGUF [hidden, ffn, experts] e.g. [768, 3072, 8]
            # → [experts, ffn, hidden] → [24576, 768]
            weights = np.transpose(weights, (2, 1, 0)).reshape(-1, weights.shape[0])
        elif "ffn_down_exps" in tensor.name:
            # GGUF [ffn, hidden, experts] e.g. [3072, 768, 8]
            # → [experts, ffn, hidden] → [24576, 768]
            weights = np.transpose(weights, (2, 0, 1)).reshape(-1, weights.shape[1])
        elif tensor.name == "token_types.weight":
            # token_type_embeddings: GGUF 1-D [768], model expects [1, 768]
            weights = weights.reshape(1, -1)
        elif weights.ndim == 2:
            # GGUF stores 2-D weights as [in, out]; PyTorch uses [out, in]
            weights = weights.T

        t = torch.from_numpy(weights.copy()).to(dtype)
        if hf_name in state_dict:
            state_dict[hf_name] = t

    model.load_state_dict(state_dict, strict=False)


class ModelVariant(StrEnum):
    """Available Nomic Embed Text v2 MoE GGUF model variants for embedding generation."""

    NOMIC_EMBED_TEXT_V2_MOE_GGUF = "nomic-embed-text-v2-moe-GGUF"


class ModelLoader(ForgeModel):
    """Nomic Embed Text v2 MoE GGUF model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.NOMIC_EMBED_TEXT_V2_MOE_GGUF: ModelConfig(
            pretrained_model_name="nomic-ai/nomic-embed-text-v2-moe-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOMIC_EMBED_TEXT_V2_MOE_GGUF

    _GGUF_FILES = {
        ModelVariant.NOMIC_EMBED_TEXT_V2_MOE_GGUF: "nomic-embed-text-v2-moe.Q4_K_M.gguf",
    }

    @property
    def GGUF_FILE(self):
        return self._GGUF_FILES[self._variant]

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
            model="Nomic-Embed-Text-v2-MoE-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            _SOURCE_REPO, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        target_dtype = dtype_override if dtype_override is not None else torch.float32

        # 1. Instantiate model architecture from the non-GGUF source repo
        model = AutoModel.from_pretrained(
            _SOURCE_REPO, trust_remote_code=True, torch_dtype=target_dtype
        )

        # 2. Download the GGUF file
        gguf_path = hf_hub_download(
            self._variant_config.pretrained_model_name, filename=self.GGUF_FILE
        )

        # 3. Load dequantized GGUF weights into the model
        _load_gguf_into_model(model, gguf_path, dtype=target_dtype)

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
