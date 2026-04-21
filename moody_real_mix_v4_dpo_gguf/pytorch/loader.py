# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Real Mix v4 DPO GGUF model loader implementation for text-to-image generation.

Loads the GGUF-quantized Lumina2Transformer2DModel from
Gthalmie1/moody-real-mix-v4-dpo-gguf, a DPO-tuned Lumina-Image-2.0 checkpoint.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import diffusers.loaders.single_file_model as _sfm
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel
from diffusers.models.normalization import LuminaLayerNormContinuous, LuminaRMSNormZero
from diffusers.models.transformers.transformer_lumina2 import (
    Lumina2CombinedTimestepCaptionEmbedding,
)

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

# This GGUF uses a larger Lumina2 variant: hidden=3840, cap_feat=2560, full MHA
IN_CHANNELS = 16
CAP_FEAT_DIM = 2560
HIDDEN_SIZE = 3840
NUM_HEADS = 30  # 3840 / 128 head_dim = 30
# Timestep MLP hidden dim: mlp.0 maps 256→1024 (so linear_1 output=1024), mlp.2 maps 1024→256.
# adaLN and norm_out conditioning however use 256-dim (GGUF weight shapes confirm this).
CONDITIONING_DIM = 1024
ACTUAL_CONDITIONING_DIM = 256  # actual dim fed into adaLN / norm_out

# Config for this model's architecture (not the default Alpha-VLLM/Lumina-Image-2.0)
_MODEL_CONFIG = {
    "_class_name": "Lumina2Transformer2DModel",
    "_diffusers_version": "0.37.0",
    "axes_dim_rope": [32, 48, 48],
    "axes_lens": [300, 512, 512],
    "cap_feat_dim": CAP_FEAT_DIM,
    # FFN: actual inner_dim = 10240 = int(8/3*3840). diffusers uses 4*dim=15360 base,
    # so ffn_dim_multiplier = 0.6667 → int(0.6667*15360)=10240, rounded to 256 → 10240
    "ffn_dim_multiplier": 0.6667,
    "hidden_size": HIDDEN_SIZE,
    "in_channels": IN_CHANNELS,
    "multiple_of": 256,
    "norm_eps": 1e-05,
    "num_attention_heads": NUM_HEADS,
    "num_kv_heads": NUM_HEADS,
    "num_layers": 30,
    "num_refiner_layers": 2,
    "out_channels": None,
    "patch_size": 2,
    "sample_size": 128,
    "scaling_factor": 1.0,
}


class ModelVariant(StrEnum):
    """Available Moody Real Mix v4 DPO GGUF model variants."""

    Q4_K_M = "Q4_K_M"


_GGUF_FILES = {
    ModelVariant.Q4_K_M: "moodyRealMix_zitV4DPO_q4_k_m.gguf",
}


def _patched_convert_lumina2_to_diffusers(checkpoint, **kwargs):
    """
    Patched version of diffusers' convert_lumina2_to_diffusers that computes
    QKV split dimensions dynamically from the tensor size instead of hardcoding
    2304/768/768.  The default Lumina2 has total QKV dim 3840 (GQA 3:1:1); this
    model uses hidden_size=3840 with full MHA (total QKV dim 11520, equal split).
    """
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

    def convert_lumina_attn_to_diffusers(tensor, diffusers_key):
        total_dim = tensor.shape[0]
        if total_dim == 3840:
            # Default Lumina2 (hidden=2304) uses GQA: q=2304, k=768, v=768
            q_dim, k_dim, v_dim = 2304, 768, 768
        else:
            # This model uses full MHA with equal Q/K/V dims
            q_dim = k_dim = v_dim = total_dim // 3
        to_q, to_k, to_v = torch.split(tensor, [q_dim, k_dim, v_dim], dim=0)
        return {
            diffusers_key.replace("qkv", "to_q"): to_q,
            diffusers_key.replace("qkv", "to_k"): to_k,
            diffusers_key.replace("qkv", "to_v"): to_v,
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
                convert_lumina_attn_to_diffusers(checkpoint.pop(key), diffusers_key)
            )
        else:
            converted_state_dict[diffusers_key] = checkpoint.pop(key)

    return converted_state_dict


@contextmanager
def _lumina2_conditioning_dim_patch():
    """
    Temporarily patch diffusers Lumina2 classes to match this model's non-standard
    timestep MLP shape: 256→1024→256 bottleneck (vs diffusers' symmetric 256→dim→dim).
    adaLN and norm_out conditioning use the 256-dim output (ACTUAL_CONDITIONING_DIM),
    not the hardcoded min(hidden_size=3840, 1024)=1024.
    """
    from diffusers.models.normalization import RMSNorm
    from diffusers.models.embeddings import Timesteps, TimestepEmbedding

    orig_rms_init = LuminaRMSNormZero.__init__
    orig_combined_init = Lumina2CombinedTimestepCaptionEmbedding.__init__
    orig_lumina2_init = Lumina2Transformer2DModel.__init__

    def _rms_init(self, embedding_dim, norm_eps, norm_elementwise_affine):
        nn.Module.__init__(self)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(ACTUAL_CONDITIONING_DIM, 4 * embedding_dim, bias=True)
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def _combined_init(
        self,
        hidden_size=4096,
        cap_feat_dim=2048,
        frequency_embedding_size=256,
        norm_eps=1e-5,
    ):
        nn.Module.__init__(self)
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        # Bottleneck MLP: 256→1024 (linear_1) then 1024→256 (linear_2, via out_dim)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size,
            time_embed_dim=CONDITIONING_DIM,
            out_dim=ACTUAL_CONDITIONING_DIM,
        )
        self.caption_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, hidden_size, bias=True),
        )

    def _lumina2_init(
        self,
        hidden_size=2304,
        cap_feat_dim=2304,
        num_attention_heads=24,
        num_kv_heads=8,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        num_layers=26,
        num_refiner_layers=2,
        in_channels=16,
        out_channels=None,
        patch_size=2,
        sample_size=128,
        scaling_factor=1.0,
        axes_dim_rope=(32, 32, 32),
        axes_lens=(300, 512, 512),
    ):
        orig_lumina2_init(
            self,
            hidden_size=hidden_size,
            cap_feat_dim=cap_feat_dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            axes_dim_rope=axes_dim_rope,
            axes_lens=axes_lens,
        )
        # Rebuild norm_out with the correct 256-dim conditioning input
        out_channels = self.out_channels
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=ACTUAL_CONDITIONING_DIM,
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * out_channels,
        )

    LuminaRMSNormZero.__init__ = _rms_init
    Lumina2CombinedTimestepCaptionEmbedding.__init__ = _combined_init
    Lumina2Transformer2DModel.__init__ = _lumina2_init

    try:
        yield
    finally:
        LuminaRMSNormZero.__init__ = orig_rms_init
        Lumina2CombinedTimestepCaptionEmbedding.__init__ = orig_combined_init
        Lumina2Transformer2DModel.__init__ = orig_lumina2_init


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

        # Patch the conversion function to handle this model's hidden_size=3840
        _sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"][
            "checkpoint_mapping_fn"
        ] = _patched_convert_lumina2_to_diffusers

        # Provide a config that matches the actual GGUF architecture, and patch
        # diffusers classes to use conditioning_dim=256 (not hardcoded min(dim,1024)=1024)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(_MODEL_CONFIG, f)

            with _lumina2_conditioning_dim_patch():
                self.transformer = Lumina2Transformer2DModel.from_single_file(
                    f"https://huggingface.co/{repo_id}/{gguf_filename}",
                    config=tmp_dir,
                    quantization_config=quantization_config,
                    torch_dtype=compute_dtype,
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
