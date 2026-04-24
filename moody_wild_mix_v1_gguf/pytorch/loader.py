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

from typing import Any, Optional

import torch
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

# This GGUF variant uses a larger Lumina2 architecture than the standard diffusers config:
#   hidden_size=3840, head_dim=128, num_heads=30, no GQA, cap_feat_dim=2560
# The timestep embedder also compresses back to FREQ_EMBED_SIZE=256 (unlike standard
# which keeps min(hidden_size, 1024)=1024), so all adaLN conditioning uses 256 dims.
HIDDEN_SIZE = 3840
NUM_HEADS = 30
NUM_LAYERS = 30
CAP_FEAT_DIM = 2560
FREQ_EMBED_SIZE = 256  # t_embedder output dim (compressed back from 1024 intermediate)
IN_CHANNELS = 16
AXES_DIM_ROPE = (32, 64, 32)  # sums to head_dim=128
FFN_DIM_MULTIPLIER = 2 / 3  # SwiGLU: inner_dim = int(4 * hidden_size * 2/3) = 10240


class ModelVariant(StrEnum):
    """Available Moody Wild Mix v1 GGUF model variants."""

    BASE_Q4_K_M = "moodyWildMix_v10Base50steps_Q4_K_M"
    DISTILLED_Q4_K_M = "moodyWildMix_v10Distilled10steps_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.BASE_Q4_K_M: "moodyWildMix_v10Base50steps_q4_k_m.gguf",
    ModelVariant.DISTILLED_Q4_K_M: "moodyWildMix_v10Distilled10steps_q4_k_m.gguf",
}


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

    @staticmethod
    def _patch_lumina2_for_gguf():
        """Patch diffusers to load this non-standard Lumina2 GGUF correctly.

        This GGUF differs from the standard Alpha-VLLM/Lumina-Image-2.0 in:
        1. hidden_size=3840, num_heads=30, head_dim=128, no GQA → qkv tensors
           have dim 11520=3*3840 requiring equal [3840,3840,3840] split, not [2304,768,768].
        2. adaLN weights stored as adaLN_modulation.0 (Linear at index 0, not index 1).
        3. timestep_embedder compresses back to 256 dims (out_dim=256), so all adaLN
           conditioning uses 256 dims instead of min(hidden_size,1024)=1024.
        """
        import torch.nn as nn
        import diffusers.loaders.single_file_model as sfm
        import diffusers.loaders.single_file_utils as sfu
        import diffusers.models.normalization as norm_mod
        import diffusers.models.transformers.transformer_lumina2 as lumina2_mod
        from diffusers.models.normalization import RMSNorm
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps

        # ── Patch 1: checkpoint conversion ───────────────────────────────────
        def patched_convert_fn(checkpoint, **kwargs):
            converted_state_dict = {}
            checkpoint.pop("norm_final.weight", None)
            keys = list(checkpoint.keys())
            for k in keys:
                if "model.diffusion_model." in k:
                    checkpoint[
                        k.replace("model.diffusion_model.", "")
                    ] = checkpoint.pop(k)

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
                # adaLN_modulation.0 = Linear at index 0 (SiLU inlined, not in Sequential)
                "adaLN_modulation.0": "norm1.linear",
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
                total = tensor.shape[0]
                # Standard Lumina2: q=2304 (24q×96), k=v=768 (8kv×96)
                if total == 2304 + 768 + 768:
                    q_dim, k_dim, v_dim = 2304, 768, 768
                elif total % 3 == 0:
                    # Equal split: no GQA, all heads same dim (e.g. 30×128=3840 each)
                    q_dim = k_dim = v_dim = total // 3
                else:
                    raise ValueError(
                        f"Cannot determine QKV split for tensor dim {total}"
                    )
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
                        convert_lumina_attn_to_diffusers(
                            checkpoint.pop(key), diffusers_key
                        )
                    )
                else:
                    converted_state_dict[diffusers_key] = checkpoint.pop(key)

            return converted_state_dict

        sfu.convert_lumina2_to_diffusers = patched_convert_fn
        sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"][
            "checkpoint_mapping_fn"
        ] = patched_convert_fn

        # ── Patch 2: LuminaRMSNormZero conditioning dim 1024→256 ─────────────
        def patched_rms_init(self, embedding_dim, norm_eps, norm_elementwise_affine):
            nn.Module.__init__(self)
            self.silu = nn.SiLU()
            self.linear = nn.Linear(FREQ_EMBED_SIZE, 4 * embedding_dim, bias=True)
            self.norm = RMSNorm(embedding_dim, eps=norm_eps)

        norm_mod.LuminaRMSNormZero.__init__ = patched_rms_init

        # ── Patch 3: timestep_embedder output dim 1024→256 ───────────────────
        def patched_temb_init(
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
            self.timestep_embedder = TimestepEmbedding(
                in_channels=frequency_embedding_size,
                time_embed_dim=min(hidden_size, 1024),
                out_dim=frequency_embedding_size,
            )
            self.caption_embedder = nn.Sequential(
                RMSNorm(cap_feat_dim, eps=norm_eps),
                nn.Linear(cap_feat_dim, hidden_size, bias=True),
            )

        lumina2_mod.Lumina2CombinedTimestepCaptionEmbedding.__init__ = patched_temb_init

        # ── Patch 4: norm_out (LuminaLayerNormContinuous) conditioning dim ────
        original_llnc_init = norm_mod.LuminaLayerNormContinuous.__init__

        def patched_llnc_init(
            self,
            embedding_dim,
            conditioning_embedding_dim,
            elementwise_affine=True,
            eps=1e-5,
            bias=True,
            norm_type="layer_norm",
            out_dim=None,
        ):
            from diffusers.models.normalization import LayerNorm

            nn.Module.__init__(self)
            self.silu = nn.SiLU()
            # Use FREQ_EMBED_SIZE=256 regardless of passed conditioning_embedding_dim
            self.linear_1 = nn.Linear(FREQ_EMBED_SIZE, embedding_dim, bias=bias)
            if norm_type == "layer_norm":
                self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
            elif norm_type == "rms_norm":
                self.norm = RMSNorm(
                    embedding_dim, eps=eps, elementwise_affine=elementwise_affine
                )
            else:
                raise ValueError(f"unknown norm_type {norm_type}")
            self.linear_2 = None
            if out_dim is not None:
                self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

        norm_mod.LuminaLayerNormContinuous.__init__ = patched_llnc_init

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the GGUF-quantized Lumina2 transformer."""
        from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel

        self._patch_lumina2_for_gguf()

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=gguf_filename,
        )

        self._transformer = Lumina2Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            num_kv_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            cap_feat_dim=CAP_FEAT_DIM,
            axes_dim_rope=AXES_DIM_ROPE,
            ffn_dim_multiplier=FFN_DIM_MULTIPLIER,
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

        # Text encoder hidden states from text encoder: (B, seq_len, cap_feat_dim)
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
