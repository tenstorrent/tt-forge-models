# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Real Mix v4 DPO GGUF model loader implementation for text-to-image generation.

Loads the GGUF-quantized Lumina2Transformer2DModel from
Gthalmie1/moody-real-mix-v4-dpo-gguf, a DPO-tuned Lumina-Image-2.0 checkpoint.

This model uses the Lumina2 5B architecture (hidden_size=3840, 30 heads, no GQA)
rather than the standard 2B (hidden_size=2304, 24Q/8KV heads). A patch is applied
to the diffusers checkpoint conversion to handle the different QKV dimensions.
"""

import torch
import diffusers.loaders.single_file_utils as _sfu
from contextlib import contextmanager
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel
from huggingface_hub import hf_hub_download
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

REPO_ID = "Gthalmie1/moody-real-mix-v4-dpo-gguf"

# Lumina2 5B architecture constants (hidden_size=3840, 30 heads, no GQA)
IN_CHANNELS = 16
HIDDEN_SIZE = 3840
NUM_LAYERS = 30
NUM_ATTENTION_HEADS = 30
NUM_KV_HEADS = 30
CAP_FEAT_DIM = 2560  # text encoder hidden size for this checkpoint


class ModelVariant(StrEnum):
    """Available Moody Real Mix v4 DPO GGUF model variants."""

    Q4_K_M = "Q4_K_M"


_GGUF_FILES = {
    ModelVariant.Q4_K_M: "moodyRealMix_zitV4DPO_q4_k_m.gguf",
}


def _patched_convert_lumina2(checkpoint, **kwargs):
    """Replaces convert_lumina2_to_diffusers with dynamic QKV splitting.

    diffusers hardcodes q=2304, k=v=768 (2B GQA). This checkpoint uses
    hidden_size=3840 with no GQA, so q=k=v=3840.
    """

    def _dynamic_attn_split(tensor, diffusers_key):
        total = tensor.shape[0]
        if total == 2304 + 768 + 768:
            q_dim, k_dim, v_dim = 2304, 768, 768
        else:
            q_dim = k_dim = v_dim = total // 3
        to_q, to_k, to_v = torch.split(tensor, [q_dim, k_dim, v_dim], dim=0)
        return {
            diffusers_key.replace("qkv", "to_q"): to_q,
            diffusers_key.replace("qkv", "to_k"): to_k,
            diffusers_key.replace("qkv", "to_v"): to_v,
        }

    converted_state_dict = {}
    checkpoint.pop("norm_final.weight", None)

    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    LUMINA_KEY_MAP = {
        "cap_embedder": "time_caption_embed.caption_embedder",
        "t_embedder.mlp.0": "time_caption_embed.timestep_embedder.linear_1",
        "t_embedder.mlp.2": "time_caption_embed.timestep_embedder.linear_2",
        "attention": "attn",
        ".out.": ".to_out.0.",
        "k_norm": "norm_k",
        "q_norm": "norm_q",
        "w1": "linear_1",
        "w2": "linear_2",
        "w3": "linear_3",
        "adaLN_modulation.1": "norm1.linear",
    }
    ATTENTION_NORM_MAP = {
        "attention_norm1": "norm1.norm",
        "attention_norm2": "norm2",
    }
    CONTEXT_REFINER_MAP = {
        "context_refiner.0.attention_norm1": "context_refiner.0.norm1",
        "context_refiner.0.attention_norm2": "context_refiner.0.norm2",
        "context_refiner.1.attention_norm1": "context_refiner.1.norm1",
        "context_refiner.1.attention_norm2": "context_refiner.1.norm2",
    }
    FINAL_LAYER_MAP = {
        "final_layer.adaLN_modulation.1": "norm_out.linear_1",
        "final_layer.linear": "norm_out.linear_2",
    }

    for key in keys:
        diffusers_key = key
        for k, v in CONTEXT_REFINER_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in FINAL_LAYER_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in ATTENTION_NORM_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in LUMINA_KEY_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)

        if "qkv" in diffusers_key:
            converted_state_dict.update(
                _dynamic_attn_split(checkpoint.pop(key), diffusers_key)
            )
        else:
            converted_state_dict[diffusers_key] = checkpoint.pop(key)

    return converted_state_dict


@contextmanager
def _patch_lumina2_conversion():
    original = _sfu.convert_lumina2_to_diffusers
    _sfu.convert_lumina2_to_diffusers = _patched_convert_lumina2
    try:
        yield
    finally:
        _sfu.convert_lumina2_to_diffusers = original


class ModelLoader(ForgeModel):
    """Moody Real Mix v4 DPO GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Moody Real Mix v4 DPO GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        repo_id = self._variant_config.pretrained_model_name
        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

        with _patch_lumina2_conversion():
            self.transformer = Lumina2Transformer2DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_attention_heads=NUM_ATTENTION_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                cap_feat_dim=CAP_FEAT_DIM,
            )
        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Latent image: (B, in_channels, H, W)
        height = 128
        width = 128
        hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)

        # Timestep: (B,)
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        # Text encoder hidden states: (B, seq_len, cap_feat_dim)
        max_sequence_length = 128
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
        )

        # Encoder attention mask: (B, seq_len)
        encoder_attention_mask = torch.ones(
            batch_size, max_sequence_length, dtype=torch.bool
        )

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }

        return inputs
