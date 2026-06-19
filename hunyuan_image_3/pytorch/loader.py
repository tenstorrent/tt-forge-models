# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage-3.0 per-component loader.

tencent/HunyuanImage-3.0 is a *unified autoregressive* native-multimodal
text-to-image model (class ``HunyuanImage3ForCausalMM``, model_type
``hunyuan_image_3_moe``, custom_code). It is the largest open-source image-gen
MoE to date: 80 B total parameters, 13 B activated per token (64 experts,
top-8, 1 shared expert), distributed in bf16 across 32 safetensor shards
(~168.5 GB on disk).

Unlike a DiT/UNet diffusion pipeline, the generator IS the MoE transformer: it
runs a flow-matching diffusion loop (``diff_infer_steps=50``) in the image-token
space, with a small UNet-style image head (patch_embed / time_embed /
final_layer) projecting between VAE latents and transformer tokens. The
checkpoint also ships a SigLIP2 vision encoder (image understanding /
conditioning) and a 3D-conv VAE (latent<->pixel).

The full 168.5 GB model does NOT fit the target device's DRAM, so it is brought
up by independently-compilable components, each scaffolded as a clean
tensors-only ``torch.nn.Module`` the runner can compile + PCC-compare in
isolation:

  * vae_decoder       AutoencoderKLConv3D.decode  (~1.26 B) latent -> pixels
  * vision_encoder    Siglip2VisionTransformer    (~0.43 B) image understanding
  * transformer_block ONE HunyuanImage3DecoderLayer (~2.5 B/layer; x32 == 80 B)
                      -- representative of the MoE backbone (the "denoiser").
  * image_head        UNetDown + time_embed + UNetUp (~0.3 B) latent<->token

Components are built from the remote config with random init (no 168 GB weight
load) via the transformers dynamic-module mechanism (trust_remote_code), the
same approach used by deepseek/pytorch/loader.py. PCC stays valid because the
runner compares CPU vs device runs of the *same* instance.

The MoE forward contains hardcoded CUDA-only side effects
(``torch.cuda.set_device`` and ``torch.cuda.nvtx.range``); these are no-ops for
compute and are neutralized inside the wrapper forward only (never globally, so
suite-wide test collection is unaffected).

NOTE: requires diffusers (VAE) and einops. See requirements.txt.
"""

import contextlib
import math
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

HUNYUAN_IMAGE_3_REPO_ID = "tencent/HunyuanImage-3.0"

DTYPE = torch.bfloat16

# ---- native-resolution-derived shape constants ------------------------------
# Native generation target is 1024x1024. VAE ffactor_spatial=16 => 64x64 latent;
# latent_channels=32; single image => 1 latent frame.
LATENT_CHANNELS = 32
LATENT_HW = 64  # 1024 / vae_ffactor_spatial(16)
LATENT_T = 1

# SigLIP2 vision encoder native input: 224px, patch16 => 256 patches, 1152 dim.
VIT_NUM_PATCHES = 256
VIT_PATCH_DIM = 3 * 16 * 16  # num_channels * patch_size^2 = 768
VIT_SIDE = 16  # isqrt(256)

# Single MoE decoder layer: hidden 4096, head_dim 128. A representative sequence
# length for op/compile validation (the full native image-token sequence is
# ~4096 tokens; op support is independent of seq_len).
BLOCK_SEQ_LEN = 512
HIDDEN_SIZE = 4096
HEAD_DIM = 128


@contextlib.contextmanager
def _cuda_shims():
    """Neutralize CUDA-only side effects in the MoE forward.

    HunyuanMoE.forward calls ``torch.cuda.set_device`` and wraps compute in
    ``torch.cuda.nvtx.range`` -- both raise on a CPU/XLA build and neither
    affects the computed result. Patched only for the duration of the wrapped
    forward so global state (and other tests during collection) is untouched.
    """
    saved_set_device = torch.cuda.set_device
    saved_nvtx_range = torch.cuda.nvtx.range
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()
    try:
        yield
    finally:
        torch.cuda.set_device = saved_set_device
        torch.cuda.nvtx.range = saved_nvtx_range


def _get_remote_class(class_reference: str):
    """Fetch a class/function from the model's custom code (trust_remote_code)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(
        class_reference, HUNYUAN_IMAGE_3_REPO_ID, trust_remote_code=True
    )


def _get_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        HUNYUAN_IMAGE_3_REPO_ID, trust_remote_code=True
    )
    # The dynamically built config leaves _attn_implementation unset; the
    # decoder layer selects its attention class from this key.
    config._attn_implementation = "sdpa"
    return config


# ----------------------------- component wrappers ----------------------------
class _VaeDecoderWrapper(torch.nn.Module):
    """AutoencoderKLConv3D.decode as forward(latent) -> image [B, 3, T, H, W]."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


class _VisionEncoderWrapper(torch.nn.Module):
    """Siglip2VisionTransformer -> last_hidden_state [B, num_patches, 1152]."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values, attention_mask, spatial_shapes):
        out = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            spatial_shapes=spatial_shapes,
        )
        return out.last_hidden_state


class _TransformerBlockWrapper(torch.nn.Module):
    """ONE HunyuanImage3DecoderLayer (MoE) -> hidden_states [B, seq, 4096].

    The 2D-RoPE (cos, sin) are structural, depend only on seq_len, and are fed
    as inputs. CUDA-only side effects in the MoE path are shimmed for the
    duration of the forward.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, cos, sin):
        with _cuda_shims():
            out = self.layer(hidden_states, custom_pos_emb=(cos, sin))
        return out[0]


class _ImageHeadWrapper(torch.nn.Module):
    """UNet-style image head: latent -> tokens -> latent.

    forward(latent, timestep): patch_embed projects the VAE latent to the
    transformer token space (conditioned on a timestep embedding), and
    final_layer projects back to a predicted latent -- exercising both the
    down-projection and up-projection ops of the diffusion image head.
    """

    def __init__(self, patch_embed, time_embed, time_embed_2, final_layer):
        super().__init__()
        self.patch_embed = patch_embed
        self.time_embed = time_embed
        self.time_embed_2 = time_embed_2
        self.final_layer = final_layer

    def forward(self, latent, timestep):
        t_emb = self.time_embed(timestep)
        image_seq, token_h, token_w = self.patch_embed(latent, t_emb)
        t_emb_2 = self.time_embed_2(timestep)
        return self.final_layer(image_seq, t_emb_2, token_h, token_w)


class ModelVariant(StrEnum):
    """Independently-compilable components of HunyuanImage-3.0."""

    VAE_DECODER = "hunyuanimage3_vae_decoder"
    VISION_ENCODER = "hunyuanimage3_vision_encoder"
    TRANSFORMER_BLOCK = "hunyuanimage3_transformer_block"
    IMAGE_HEAD = "hunyuanimage3_image_head"


class ModelLoader(ForgeModel):
    """Per-component loader for HunyuanImage-3.0.

    load_model() returns just the requested component (wrapped to a clean
    tensors-only forward, random init). load_inputs() builds synthetic tensors
    at the native-resolution-derived shapes. The full 80 B pipeline is never
    instantiated.
    """

    _VARIANTS = {
        ModelVariant.VAE_DECODER: ModelConfig(
            pretrained_model_name=HUNYUAN_IMAGE_3_REPO_ID
        ),
        ModelVariant.VISION_ENCODER: ModelConfig(
            pretrained_model_name=HUNYUAN_IMAGE_3_REPO_ID
        ),
        ModelVariant.TRANSFORMER_BLOCK: ModelConfig(
            pretrained_model_name=HUNYUAN_IMAGE_3_REPO_ID
        ),
        ModelVariant.IMAGE_HEAD: ModelConfig(
            pretrained_model_name=HUNYUAN_IMAGE_3_REPO_ID
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER_BLOCK

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.CV_IMAGE_FE
            if variant == ModelVariant.VISION_ENCODER
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="HunyuanImage3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else DTYPE
        config = _get_config()

        if self._variant == ModelVariant.VAE_DECODER:
            AutoencoderKLConv3D = _get_remote_class(
                "autoencoder_kl_3d.AutoencoderKLConv3D"
            )
            vae = AutoencoderKLConv3D.from_config(config.vae).to(dtype).eval()
            return _VaeDecoderWrapper(vae)

        if self._variant == ModelVariant.VISION_ENCODER:
            Siglip2VisionTransformer = _get_remote_class(
                "siglip2.Siglip2VisionTransformer"
            )
            vit = Siglip2VisionTransformer(config.vit).to(dtype).eval()
            return _VisionEncoderWrapper(vit)

        if self._variant == ModelVariant.TRANSFORMER_BLOCK:
            HunyuanImage3DecoderLayer = _get_remote_class(
                "hunyuan.HunyuanImage3DecoderLayer"
            )
            layer = HunyuanImage3DecoderLayer(config, layer_idx=0).to(dtype).eval()
            return _TransformerBlockWrapper(layer)

        if self._variant == ModelVariant.IMAGE_HEAD:
            UNetDown = _get_remote_class("hunyuan.UNetDown")
            UNetUp = _get_remote_class("hunyuan.UNetUp")
            TimestepEmbedder = _get_remote_class("hunyuan.TimestepEmbedder")
            h = config.hidden_size
            patch_embed = UNetDown(
                patch_size=config.patch_size,
                in_channels=config.vae["latent_channels"],
                emb_channels=h,
                hidden_channels=config.patch_embed_hidden_dim,
                out_channels=h,
            )
            final_layer = UNetUp(
                patch_size=config.patch_size,
                in_channels=h,
                emb_channels=h,
                hidden_channels=config.patch_embed_hidden_dim,
                out_channels=config.vae["latent_channels"],
                out_norm=True,
            )
            time_embed = TimestepEmbedder(hidden_size=h)
            time_embed_2 = TimestepEmbedder(hidden_size=h)
            head = _ImageHeadWrapper(
                patch_embed, time_embed, time_embed_2, final_layer
            )
            return head.to(dtype).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        """Return synthetic inputs matching each component's forward signature."""
        dtype = dtype_override if dtype_override is not None else DTYPE
        B = batch_size

        if self._variant == ModelVariant.VAE_DECODER:
            latent = torch.randn(
                B, LATENT_CHANNELS, LATENT_T, LATENT_HW, LATENT_HW, dtype=dtype
            )
            return [latent]

        if self._variant == ModelVariant.VISION_ENCODER:
            pixel_values = torch.randn(B, VIT_NUM_PATCHES, VIT_PATCH_DIM, dtype=dtype)
            attention_mask = torch.ones(B, VIT_NUM_PATCHES, dtype=torch.long)
            spatial_shapes = torch.tensor(
                [[VIT_SIDE, VIT_SIDE]] * B, dtype=torch.long
            )
            return [pixel_values, attention_mask, spatial_shapes]

        if self._variant == ModelVariant.TRANSFORMER_BLOCK:
            hidden_states = torch.randn(B, BLOCK_SEQ_LEN, HIDDEN_SIZE, dtype=dtype)
            build_batch_2d_rope = _get_remote_class("hunyuan.build_batch_2d_rope")
            cos, sin = build_batch_2d_rope(seq_len=BLOCK_SEQ_LEN, n_elem=HEAD_DIM)
            cos = cos.to(dtype).expand(B, -1, -1).contiguous()
            sin = sin.to(dtype).expand(B, -1, -1).contiguous()
            return [hidden_states, cos, sin]

        if self._variant == ModelVariant.IMAGE_HEAD:
            latent = torch.randn(
                B, LATENT_CHANNELS, LATENT_HW, LATENT_HW, dtype=dtype
            )
            timestep = torch.randint(0, 1000, (B,)).to(dtype)
            return [latent, timestep]

        raise ValueError(f"Unknown variant: {self._variant}")
