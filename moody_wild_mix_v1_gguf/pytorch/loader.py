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
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

REPO_ID = "Gthalmie1/moody-wild-mix-v1-gguf"

# Actual architecture constants for this model variant (larger than reference Lumina-Image-2.0)
HIDDEN_SIZE = 3840
NUM_ATTENTION_HEADS = 30  # head_dim = HIDDEN_SIZE / NUM_ATTENTION_HEADS = 128
NUM_KV_HEADS = 30  # MHA (equal to NUM_ATTENTION_HEADS)
CAP_FEAT_DIM = 2560  # text encoder hidden size
NUM_LAYERS = 30
NUM_REFINER_LAYERS = 2
FFN_DIM_MULTIPLIER = 2 / 3  # inner_dim = int(2/3 * 4 * HIDDEN_SIZE) = 10240
TIME_EMBED_DIM_OUT = (
    256  # t_embedder output dim (differs from diffusers default of min(hidden,1024))
)
IN_CHANNELS = 16
PATCH_SIZE = 2


class ModelVariant(StrEnum):
    """Available Moody Wild Mix v1 GGUF model variants."""

    BASE_Q4_K_M = "moodyWildMix_v10Base50steps_Q4_K_M"
    DISTILLED_Q4_K_M = "moodyWildMix_v10Distilled10steps_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.BASE_Q4_K_M: "moodyWildMix_v10Base50steps_q4_k_m.gguf",
    ModelVariant.DISTILLED_Q4_K_M: "moodyWildMix_v10Distilled10steps_q4_k_m.gguf",
}


def _make_lumina2_convert_fn():
    """Build a patched convert_lumina2_to_diffusers that infers QKV split from tensor shape."""
    import torch
    from diffusers.loaders.single_file_utils import (
        convert_lumina2_to_diffusers as _orig,
    )

    def patched_convert(checkpoint, **kwargs):
        # Run the original conversion but intercept the QKV split error.
        # The original hardcodes [2304, 768, 768] which is wrong for this larger variant.
        # We patch the inner closure by re-implementing the conversion.
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

        def convert_qkv(tensor, diffusers_key):
            total = tensor.shape[0]
            # Split evenly for MHA (q=k=v), fall back to original for GQA
            if total % 3 == 0:
                d = total // 3
                to_q, to_k, to_v = torch.split(tensor, [d, d, d], dim=0)
            else:
                to_q, to_k, to_v = torch.split(tensor, [2304, 768, 768], dim=0)
            return {
                diffusers_key.replace("qkv", "to_q"): to_q,
                diffusers_key.replace("qkv", "to_k"): to_k,
                diffusers_key.replace("qkv", "to_v"): to_v,
            }

        # Remap adaLN_modulation.0 → norm1.linear (GGUF uses .0 index, diffusers uses .1)
        keys = list(checkpoint.keys())
        for k in keys:
            if "adaLN_modulation.0" in k:
                new_k = k.replace("adaLN_modulation.0", "adaLN_modulation.1")
                checkpoint[new_k] = checkpoint.pop(k)

        keys = list(checkpoint.keys())
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
                    convert_qkv(checkpoint.pop(key), diffusers_key)
                )
            else:
                converted_state_dict[diffusers_key] = checkpoint.pop(key)

        return converted_state_dict

    return patched_convert


@contextmanager
def _patch_lumina2_loading():
    """Context manager that monkey-patches diffusers to support this model's architecture."""
    import diffusers.loaders.single_file_model as sfm
    import diffusers.models.normalization as norm_module
    import diffusers.models.transformers.transformer_lumina2 as lumina_module
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from diffusers.models.normalization import LuminaLayerNormContinuous, RMSNorm

    orig_mapping = sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"][
        "checkpoint_mapping_fn"
    ]
    orig_norm_init = norm_module.LuminaRMSNormZero.__init__
    orig_embed_init = lumina_module.Lumina2CombinedTimestepCaptionEmbedding.__init__
    orig_layer_norm_init = norm_module.LuminaLayerNormContinuous.__init__

    # 1. Patch conversion function (fixes hardcoded QKV split)
    sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"][
        "checkpoint_mapping_fn"
    ] = _make_lumina2_convert_fn()

    # 2. Patch LuminaRMSNormZero to accept TIME_EMBED_DIM_OUT as adaLN input
    def _patched_norm_init(self, embedding_dim, norm_eps, norm_elementwise_affine):
        nn.Module.__init__(self)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(TIME_EMBED_DIM_OUT, 4 * embedding_dim, bias=True)
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    norm_module.LuminaRMSNormZero.__init__ = _patched_norm_init

    # 3. Patch Lumina2CombinedTimestepCaptionEmbedding to output TIME_EMBED_DIM_OUT
    def _patched_embed_init(
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
            out_dim=TIME_EMBED_DIM_OUT,
        )
        self.caption_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, hidden_size, bias=True),
        )

    lumina_module.Lumina2CombinedTimestepCaptionEmbedding.__init__ = _patched_embed_init

    # 4. Patch LuminaLayerNormContinuous to use TIME_EMBED_DIM_OUT for conditioning dim
    # (used in norm_out of Lumina2Transformer2DModel which uses min(hidden_size, 1024))
    def _patched_layer_norm_init(
        self,
        embedding_dim,
        conditioning_embedding_dim,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim=None,
    ):
        orig_layer_norm_init(
            self,
            embedding_dim=embedding_dim,
            conditioning_embedding_dim=TIME_EMBED_DIM_OUT,
            elementwise_affine=elementwise_affine,
            eps=eps,
            bias=bias,
            norm_type=norm_type,
            out_dim=out_dim,
        )

    norm_module.LuminaLayerNormContinuous.__init__ = _patched_layer_norm_init

    try:
        yield
    finally:
        sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"][
            "checkpoint_mapping_fn"
        ] = orig_mapping
        norm_module.LuminaRMSNormZero.__init__ = orig_norm_init
        lumina_module.Lumina2CombinedTimestepCaptionEmbedding.__init__ = orig_embed_init
        norm_module.LuminaLayerNormContinuous.__init__ = orig_layer_norm_init


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

        gguf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_GGUF_FILES[self._variant],
        )

        with _patch_lumina2_loading():
            self._transformer = Lumina2Transformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
                hidden_size=HIDDEN_SIZE,
                num_attention_heads=NUM_ATTENTION_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                cap_feat_dim=CAP_FEAT_DIM,
                num_layers=NUM_LAYERS,
                num_refiner_layers=NUM_REFINER_LAYERS,
                ffn_dim_multiplier=FFN_DIM_MULTIPLIER,
                # axes_dim_rope must sum to head_dim (HIDDEN_SIZE / NUM_ATTENTION_HEADS = 128)
                axes_dim_rope=(32, 48, 48),
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
