# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Wild Mix v1 GGUF (Gthalmie1/moody-wild-mix-v1-gguf) model loader implementation.

Moody Wild Mix v1 is a text-to-image generation model in GGUF quantized format,
based on the Lumina-Image-2.0 architecture (~6B parameters). The original weights
come from the CivitAI "Moody Wild Mix" model by catlover1937.

Available variants:
- BASE_Q4_K_M: Base 50-step Q4_K_M quantized transformer
- DISTILLED_Q4_K_M: Distilled 10-step Q4_K_M quantized transformer
"""

from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

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

REPO_ID = "Gthalmie1/moody-wild-mix-v1-gguf"

# Architecture constants for this GGUF variant (differs from reference Lumina-Image-2.0):
# - hidden_size=3840, MHA 30 heads (head_dim=128), cap_feat_dim=2560
# - Timestep bottleneck MLP: 256->1024->256 (output 256, not 1024)
# - AdaLN conditioning dim: 256 (matches bottleneck output), not min(dim, 1024)
# - AdaLN sequential index: .0 (not .1 as in standard Lumina2)
IN_CHANNELS = 16
CAP_FEAT_DIM = 2560  # text encoder output dim (cap_embedder input size)
_TIMESTEP_EMBED_DIM = 256  # timestep bottleneck output dim


class ModelVariant(StrEnum):
    """Available Moody Wild Mix v1 GGUF model variants."""

    BASE_Q4_K_M = "moodyWildMix_v10Base50steps_Q4_K_M"
    DISTILLED_Q4_K_M = "moodyWildMix_v10Distilled10steps_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.BASE_Q4_K_M: "moodyWildMix_v10Base50steps_q4_k_m.gguf",
    ModelVariant.DISTILLED_Q4_K_M: "moodyWildMix_v10Distilled10steps_q4_k_m.gguf",
}


@contextmanager
def _patch_lumina2_for_gguf():
    """Patch diffusers Lumina2 classes to load this GGUF variant.

    This GGUF uses a Lumina2 architectural variant that differs from the
    reference model supported by diffusers in three ways:

    1. QKV split: MHA equal-thirds (11520 → 3×3840) vs GQA (3840 → 2304+768+768)
    2. Timestep embedding: 256→1024→256 bottleneck (output 256) vs 256→1024 (output 1024)
    3. AdaLN conditioning dim: 256 (from bottleneck) vs min(hidden_size, 1024)=1024
    4. AdaLN sequential key: .0 (linear at index 0) vs .1 (SiLU at 0, linear at 1)

    All patches are applied via the context manager and restored afterward.
    """
    import diffusers.loaders.single_file_model as _sfm
    import diffusers.models.normalization as _norm
    import diffusers.models.transformers.transformer_lumina2 as _tl2
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from diffusers.models.normalization import RMSNorm

    # --- Patch 1: checkpoint converter ---
    _CLASS_ENTRY = _sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"]
    _orig_converter = _CLASS_ENTRY["checkpoint_mapping_fn"]

    def _patched_converter(checkpoint, **kwargs):
        converted_state_dict = {}
        checkpoint.pop("norm_final.weight", None)
        keys = list(checkpoint.keys())
        for k in keys:
            if "model.diffusion_model." in k:
                checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

        # Standard key renames — NOTE: adaLN uses .0 (not .1) in this GGUF variant
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
            "adaLN_modulation.0": "norm1.linear",
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
        # final_layer uses .1 (not .0) for its adaLN
        FINAL_LAYER_MAP = {
            "final_layer.adaLN_modulation.1": "norm_out.linear_1",
            "final_layer.linear": "norm_out.linear_2",
        }

        def _split_qkv(tensor, diffusers_key):
            total_dim = tensor.shape[0]
            # Standard GQA: q=2304, k=v=768 (sum=3840); MHA variant: equal thirds
            if total_dim == 2304 + 768 + 768:
                q_dim, k_dim, v_dim = 2304, 768, 768
            else:
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
                    _split_qkv(checkpoint.pop(key), diffusers_key)
                )
            else:
                converted_state_dict[diffusers_key] = checkpoint.pop(key)
        return converted_state_dict

    _CLASS_ENTRY["checkpoint_mapping_fn"] = _patched_converter

    # --- Patch 2: LuminaRMSNormZero — use 256-dim conditioning input ---
    _orig_rms_init = _norm.LuminaRMSNormZero.__init__

    def _patched_rms_init(self, embedding_dim, norm_eps, norm_elementwise_affine):
        nn.Module.__init__(self)
        # This GGUF variant feeds a 256-dim timestep embedding directly to each adaLN
        # (not the standard min(embedding_dim, 1024)).
        self.linear = nn.Linear(_TIMESTEP_EMBED_DIM, 4 * embedding_dim, bias=True)
        self.norm = RMSNorm(
            embedding_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )

    _norm.LuminaRMSNormZero.__init__ = _patched_rms_init

    # --- Patch 3: Lumina2CombinedTimestepCaptionEmbedding — bottleneck timestep MLP ---
    _orig_tsembed_init = _tl2.Lumina2CombinedTimestepCaptionEmbedding.__init__

    def _patched_tsembed_init(self, hidden_size=4096, cap_feat_dim=2048, norm_eps=1e-6):
        nn.Module.__init__(self)
        freq_dim = 256
        # This GGUF variant uses a bottleneck MLP: freq_dim → 4*freq_dim → freq_dim
        self.time_proj = Timesteps(
            num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=freq_dim,
            time_embed_dim=4 * freq_dim,
            out_dim=freq_dim,
        )
        self.caption_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, hidden_size, bias=True),
        )

    _tl2.Lumina2CombinedTimestepCaptionEmbedding.__init__ = _patched_tsembed_init

    # --- Patch 4: LuminaLayerNormContinuous — use 256-dim conditioning for norm_out ---
    _orig_llnc_init = _norm.LuminaLayerNormContinuous.__init__

    def _patched_llnc_init(self, embedding_dim, conditioning_embedding_dim, **kwargs):
        # When this model's norm_out is initialized with conditioning_embedding_dim=1024
        # (from min(hidden_size, 1024)), override it to 256.
        if conditioning_embedding_dim > _TIMESTEP_EMBED_DIM:
            conditioning_embedding_dim = _TIMESTEP_EMBED_DIM
        _orig_llnc_init(self, embedding_dim, conditioning_embedding_dim, **kwargs)

    _norm.LuminaLayerNormContinuous.__init__ = _patched_llnc_init

    try:
        yield
    finally:
        _CLASS_ENTRY["checkpoint_mapping_fn"] = _orig_converter
        _norm.LuminaRMSNormZero.__init__ = _orig_rms_init
        _tl2.Lumina2CombinedTimestepCaptionEmbedding.__init__ = _orig_tsembed_init
        _norm.LuminaLayerNormContinuous.__init__ = _orig_llnc_init


class ModelLoader(ForgeModel):
    """Moody Wild Mix v1 GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BASE_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.DISTILLED_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Moody_Wild_Mix_v1_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the GGUF-quantized Lumina2 transformer."""
        from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=gguf_filename,
        )

        with _patch_lumina2_for_gguf():
            self._transformer = Lumina2Transformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
                # Architecture overrides for this GGUF variant
                hidden_size=3840,
                num_layers=30,
                num_refiner_layers=2,
                num_attention_heads=30,
                num_kv_heads=30,
                multiple_of=256,
                ffn_dim_multiplier=2 / 3,
                cap_feat_dim=CAP_FEAT_DIM,
            )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1) -> Any:
        """Prepare synthetic transformer inputs for the Moody Wild Mix v1 GGUF model."""
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Latent image: (B, in_channels, H, W)
        height = 128
        width = 128
        hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)

        # Timestep: (B,)
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        # Text encoder hidden states: (B, seq_len, cap_feat_dim)
        # This model uses cap_feat_dim=2560 (not Gemma-2-2b's 2304)
        max_sequence_length = 128
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
        )

        # Encoder attention mask: (B, seq_len)
        encoder_attention_mask = torch.ones(
            batch_size, max_sequence_length, dtype=torch.bool
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
