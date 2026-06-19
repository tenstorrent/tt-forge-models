# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage-3.0 per-component loader.

tencent/HunyuanImage-3.0 is NOT a diffusers-style pipeline; it is a single
custom-code transformers model (``HunyuanImage3ForCausalMM``) that natively
unifies text understanding and image generation in one ~80B-parameter,
64-expert Mixture-of-Experts autoregressive transformer. There is no separate
UNet/DiT denoiser: the MoE transformer *is* the generator, emitting image
tokens autoregressively which are then turned into pixels by a 3D VAE. A
SigLIP2 NaFlex vision tower + an MLP aligner provide image conditioning
(image-to-image / instruct modes).

Brought up by composite component (see the model-bringup diffusion guidance).
Every weight lives in one 168.5 GB (bf16) checkpoint sharded across 32
safetensors files -- there are no per-component subfolders, so components are
not independently downloadable the way a diffusers pipeline's are.

Components / variants
---------------------
  VisionEncoder -> SigLIP2-so400m-patch16-naflex vision tower   params ~0.43 B
                   (image conditioning; runnable + on-device validated)
  Transformer   -> HunyuanImage3ForCausalMM MoE generator        params ~80 B
                   (the compute-dominant "denoiser" equivalent)
  Vae           -> AutoencoderKLConv3D decoder (latent -> pixels) params ~0.25 B

Capacity note (target device: qb2-blackhole = 4x Blackhole p150, 32 GB/chip,
128 GB total DRAM)
------------------------------------------------------------------------------
The full checkpoint is 168.5 GB in bf16 -- larger than the 128 GB of aggregate
DRAM on the whole 4-chip QuietBox, so the MoE generator (the "denoiser") does
not fit even sharded across all four chips, and never will with realistic
activation/KV-cache headroom. Per the diffusion-decomposition rules a denoiser
that cannot run on device makes the composite bringup HW_STATUS=FAILED.

Only the SigLIP2 vision tower fits a single chip and is validated here. The
Transformer and Vae components load the genuine weights via
``trust_remote_code`` from the real repo, but are gated behind the
``HUNYUAN_IMAGE_3_ALLOW_FULL_LOAD`` env var so a CI sweep does not accidentally
pull the 168.5 GB checkpoint onto a device that cannot hold it. Set that var on
adequate hardware to exercise them.

The VisionEncoder uses the upstream ``google/siglip2-so400m-patch16-naflex``
weights as the architecture proxy for HunyuanImage-3.0's vision tower (the
config is identical -- hidden 1152, 27 layers, 16 heads, patch 16, 256 patches;
HunyuanImage's own fine-tuned vision weights are interleaved in the 168.5 GB
checkpoint and not separately downloadable).
"""

import os
from typing import Optional

import torch

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

# Repos
HUNYUAN_IMAGE_3_REPO_ID = "tencent/HunyuanImage-3.0"
# Architecture proxy for the SigLIP2 NaFlex vision tower (config-identical).
SIGLIP2_VISION_REPO_ID = "google/siglip2-so400m-patch16-naflex"

DTYPE = torch.bfloat16
# The SigLIP2 vision tower needs fp32 to clear the 0.99 PCC bar on device
# (bf16 lands ~0.95); the big MoE generator stays bf16 as shipped.
VISION_ENCODER_DTYPE = torch.float32

# Env gate: the Transformer / Vae components require the full 168.5 GB
# checkpoint and do not fit qb2-blackhole; only load them when explicitly asked.
_ALLOW_FULL_LOAD_ENV = "HUNYUAN_IMAGE_3_ALLOW_FULL_LOAD"

# ---- SigLIP2 NaFlex vision-tower shape constants (from config + captured I/O) -
VIT_NUM_PATCHES = 256  # config.vit.num_patches
VIT_PATCH_SIZE = 16
VIT_NUM_CHANNELS = 3
VIT_PATCH_EMBED_DIM = VIT_NUM_CHANNELS * VIT_PATCH_SIZE * VIT_PATCH_SIZE  # 768
VIT_HIDDEN = 1152
VIT_GRID_H = 16  # 16 x 16 = 256 valid patches for a 224/16 grid
VIT_GRID_W = 16

# ---- Transformer (MoE generator) shape constants (from config.json) ----------
TR_VOCAB_SIZE = 133120
TR_HIDDEN = 4096
TR_SEQ_LEN = 32  # short representative prefill (scaffold only)

# ---- VAE (AutoencoderKLConv3D) shape constants (from config.vae) --------------
VAE_LATENT_CHANNELS = 32
VAE_T = 1  # single image frame
VAE_Z_H = 64
VAE_Z_W = 64


class _Siglip2VisionWrapper(torch.nn.Module):
    """Adapt Siglip2VisionModel to a tensors-only forward returning the last
    hidden state [B, num_patches, 1152] that conditions the generator."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):
        out = self.vision_model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        return out.last_hidden_state


class _HunyuanImage3VaeDecoderWrapper(torch.nn.Module):
    """Expose the 3D VAE's decode(latent) -> pixels as a plain forward."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        out = self.vae.decode(latent)
        # AutoencoderKLConv3D.decode returns a DecoderOutput-like object.
        return getattr(out, "sample", out)


def _require_full_load(component: str):
    if not os.environ.get(_ALLOW_FULL_LOAD_ENV):
        raise RuntimeError(
            f"HunyuanImage-3.0 '{component}' requires the full 168.5 GB bf16 "
            f"checkpoint ({HUNYUAN_IMAGE_3_REPO_ID}). The ~80B MoE generator "
            f"does not fit qb2-blackhole (128 GB total DRAM across 4 Blackhole "
            f"chips), so this component is gated off by default. Set "
            f"{_ALLOW_FULL_LOAD_ENV}=1 on hardware with enough memory to load it."
        )


class ModelVariant(StrEnum):
    """Loadable components of the HunyuanImage-3.0 model."""

    VISION_ENCODER = "VisionEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Per-component loader for HunyuanImage-3.0.

    ``load_model`` returns just the requested component wrapped to a clean
    tensors-only forward; ``load_inputs`` builds synthetic tensors at the
    component's captured/derived shapes. The full unified model is never
    instantiated as one graph.
    """

    _VARIANTS = {
        ModelVariant.VISION_ENCODER: ModelConfig(
            pretrained_model_name=SIGLIP2_VISION_REPO_ID
        ),
        ModelVariant.TRANSFORMER: ModelConfig(
            pretrained_model_name=HUNYUAN_IMAGE_3_REPO_ID
        ),
        ModelVariant.VAE: ModelConfig(
            pretrained_model_name=HUNYUAN_IMAGE_3_REPO_ID
        ),
    }
    DEFAULT_VARIANT = ModelVariant.VISION_ENCODER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        if variant == ModelVariant.VISION_ENCODER:
            task = ModelTask.CV_IMAGE_FE
        elif variant == ModelVariant.TRANSFORMER:
            task = ModelTask.MM_IMAGE_TTT
        else:  # VAE
            task = ModelTask.CONDITIONAL_GENERATION
        return ModelInfo(
            model="HunyuanImage3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        VISION_ENCODER -> SigLIP2 NaFlex vision tower wrapper (runnable)
        TRANSFORMER    -> HunyuanImage3ForCausalMM MoE generator (gated; >device)
        VAE            -> AutoencoderKLConv3D decoder wrapper   (gated; in ckpt)
        """
        repo = self._variant_config.pretrained_model_name

        if self._variant == ModelVariant.VISION_ENCODER:
            from transformers import Siglip2VisionModel

            # The SigLIP2 tower is natively fp32 in HunyuanImage-3.0
            # (config.vit.torch_dtype == "float32"); bf16 drops device PCC to
            # ~0.95, so honor the native precision and ignore a bf16 override.
            vision_model = Siglip2VisionModel.from_pretrained(
                repo, torch_dtype=VISION_ENCODER_DTYPE
            )
            return _Siglip2VisionWrapper(vision_model.eval())

        if self._variant == ModelVariant.TRANSFORMER:
            _require_full_load("Transformer")
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                repo, trust_remote_code=True, torch_dtype=dtype
            )
            return model.eval()

        if self._variant == ModelVariant.VAE:
            _require_full_load("Vae")
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                repo, trust_remote_code=True, torch_dtype=dtype
            )
            # The 3D VAE is a sub-module of the unified model.
            vae = getattr(model, "vae", None)
            if vae is None:
                raise AttributeError(
                    "Could not locate the VAE sub-module on HunyuanImage3ForCausalMM"
                )
            return _HunyuanImage3VaeDecoderWrapper(vae.eval())

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        VISION_ENCODER -> [pixel_values (1,256,768) float,
                           pixel_attention_mask (1,256) int64,
                           spatial_shapes (1,2) int64]
        TRANSFORMER    -> [input_ids (1,32) int64, attention_mask (1,32) int64]
        VAE            -> [latent (1,32,1,64,64) float]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.VISION_ENCODER:
            # Match the tower's native fp32 (see load_model); ignore bf16 override.
            pixel_values = torch.randn(
                1, VIT_NUM_PATCHES, VIT_PATCH_EMBED_DIM, dtype=VISION_ENCODER_DTYPE
            )
            pixel_attention_mask = torch.ones(1, VIT_NUM_PATCHES, dtype=torch.long)
            spatial_shapes = torch.tensor([[VIT_GRID_H, VIT_GRID_W]], dtype=torch.long)
            return [pixel_values, pixel_attention_mask, spatial_shapes]

        if self._variant == ModelVariant.TRANSFORMER:
            input_ids = torch.randint(0, TR_VOCAB_SIZE, (1, TR_SEQ_LEN), dtype=torch.long)
            attention_mask = torch.ones(1, TR_SEQ_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.VAE:
            latent = torch.randn(
                1, VAE_LATENT_CHANNELS, VAE_T, VAE_Z_H, VAE_Z_W, dtype=dtype
            )
            return [latent]

        raise ValueError(f"Unknown variant: {self._variant}")
