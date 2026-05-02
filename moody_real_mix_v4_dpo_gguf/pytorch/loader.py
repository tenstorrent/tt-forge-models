# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Real Mix v4 DPO GGUF model loader implementation for text-to-image generation.

Loads the GGUF-quantized Lumina2Transformer2DModel from
Gthalmie1/moody-real-mix-v4-dpo-gguf, a DPO-tuned Lumina-Image-2.0 checkpoint.
"""

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel
from huggingface_hub import hf_hub_download

# GGUFParameter.as_tensor() calls _make_subclass which re-dispatches through
# __torch_function__ on the subclass, causing infinite recursion under Dynamo.
# DisableTorchFunctionSubclass breaks the cycle.
from diffusers.quantizers.gguf.utils import GGUFParameter as _GGUFParameter


def _patched_as_tensor(self):
    with torch.DisableTorchFunctionSubclass():
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)


_GGUFParameter.as_tensor = _patched_as_tensor

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

# Architecture constants derived from GGUF tensor shapes:
# - hidden_size=3840, MHA 30 heads (head_dim=128, equal-thirds QKV)
# - cap_feat_dim=2560 (cap_embedder.0 norm dim in GGUF)
# - Timestep bottleneck: 256→1024→256 (output 256)
# - AdaLN conditioning dim: 256
IN_CHANNELS = 16
CAP_FEAT_DIM = 2560
_TIMESTEP_EMBED_DIM = 256


class ModelVariant(StrEnum):
    """Available Moody Real Mix v4 DPO GGUF model variants."""

    Q4_K_M = "Q4_K_M"


_GGUF_FILES = {
    ModelVariant.Q4_K_M: "moodyRealMix_zitV4DPO_q4_k_m.gguf",
}


@contextmanager
def _patch_lumina2_for_gguf():
    """Patch diffusers Lumina2 classes to load this GGUF variant.

    This GGUF uses a Lumina2 architectural variant that differs from the
    reference model supported by diffusers in four ways:

    1. QKV split: MHA equal-thirds (11520 → 3×3840) vs GQA (3840 → 2304+768+768)
    2. Timestep embedding: 256→1024→256 bottleneck (output 256) vs 256→1024 (output 1024)
    3. AdaLN conditioning dim: 256 (from bottleneck) vs min(hidden_size, 1024)=1024
    4. AdaLN sequential key: .0 (linear at index 0) vs .1 (SiLU at 0, linear at 1)
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
        FINAL_LAYER_MAP = {
            "final_layer.adaLN_modulation.1": "norm_out.linear_1",
            "final_layer.linear": "norm_out.linear_2",
        }

        def _split_qkv(tensor, diffusers_key):
            total_dim = tensor.shape[0]
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
        self.silu = nn.SiLU()
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

        model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)
        with _patch_lumina2_for_gguf():
            self.transformer = Lumina2Transformer2DModel.from_single_file(
                model_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
                # Architecture overrides for this GGUF variant (hidden_size=3840, MHA)
                hidden_size=3840,
                num_layers=30,
                num_refiner_layers=2,
                num_attention_heads=30,
                num_kv_heads=30,
                multiple_of=256,
                ffn_dim_multiplier=2 / 3,
                cap_feat_dim=CAP_FEAT_DIM,
                axes_dim_rope=(32, 48, 48),
            )
        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        height = 128
        width = 128
        hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)

        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        max_sequence_length = 128
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
        )

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
