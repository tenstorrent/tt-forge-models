# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RenderFormer model loader implementation.

RenderFormer is a transformer-based neural renderer for triangle meshes
with global illumination. Reference: https://github.com/microsoft/renderformer

Requires the RenderFormer repository to be cloned at /tmp/renderformer_repo.
"""

import os
import sys

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel

RENDERFORMER_REPO_PATH = "/tmp/renderformer_repo"


def _ensure_renderformer_importable():
    """Ensure the RenderFormer repo is cloned and importable."""
    if not os.path.isdir(RENDERFORMER_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/microsoft/renderformer.git",
                RENDERFORMER_REPO_PATH,
            ]
        )

    if RENDERFORMER_REPO_PATH not in sys.path:
        sys.path.insert(0, RENDERFORMER_REPO_PATH)

    for key in list(sys.modules.keys()):
        if key == "renderformer" or key.startswith("renderformer."):
            del sys.modules[key]

    os.environ.setdefault("ATTN_IMPL", "sdpa")


def _patch_swin_attn_mask_device(attn_module):
    """Patch get_swin_attn_mask to include device in cache key.

    The upstream cache uses (H, W, window_size, shift_size) as key, ignoring
    device. When running under XLA/torch.compile, the mask may be cached on
    CPU and reused for xla:0 tensors, causing a device mismatch.
    """
    _orig_get_mask = attn_module.get_swin_attn_mask

    def _device_aware_get_mask(H, W, window_size, shift_size, device):
        attn_module.SWIN_ATTN_MASK_CACHE.clear()
        mask = _orig_get_mask(H, W, window_size, shift_size, device)
        return mask.to(device)

    attn_module.get_swin_attn_mask = _device_aware_get_mask


def _patch_view_transformer_autocast(view_transformer_cls):
    """Patch ViewTransformer.forward to use CPU-compatible autocast.

    The upstream code hardcodes device_type='cuda' in torch.autocast which
    fails on non-CUDA systems. Replace with a nullcontext since we run on CPU.
    """
    import contextlib
    from einops import rearrange

    _orig_forward = view_transformer_cls.forward

    def _patched_forward(
        self, camera_o, ray_map, tri_tokens, tri_pos, valid_mask, tf32_mode=False
    ):
        ray_map = self.vdir_pe(ray_map)
        ray_tokens = rearrange(
            ray_map,
            "b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)",
            p1=self.config.patch_size,
            p2=self.config.patch_size,
        )
        patch_h = ray_map.size(1) // self.config.patch_size
        patch_w = ray_map.size(2) // self.config.patch_size
        ray_tokens = self.ray_map_patch_token + self.ray_map_encoder_norm(
            self.ray_map_encoder(ray_tokens)
        )
        n_patches = ray_tokens.size(1)
        ray_token_pos = camera_o[:, None].repeat(1, n_patches, 3)

        if self.config.pe_type == "nerf":
            ray_tokens = ray_tokens + self.token_pos_pe_norm(
                self.pe_token_proj(self.pos_pe(ray_token_pos))
            )
            tri_tokens = tri_tokens + self.token_pos_pe_norm(
                self.pe_token_proj(self.pos_pe(tri_pos))
            )

        if self.config.use_dpt_decoder:
            with contextlib.nullcontext():
                out_features = self.transformer(
                    ray_tokens,
                    tri_tokens,
                    src_key_padding_mask=valid_mask,
                    triangle_pos=tri_pos,
                    ray_pos=ray_token_pos,
                    out_layers=self.out_layers,
                    tf32_mode=tf32_mode,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
            decoded_img = self.out_dpt(
                out_features, patch_h, patch_w, patch_size=self.config.patch_size
            )
            return self.out_proj_act(decoded_img)
        else:
            seq = self.transformer(
                ray_tokens,
                tri_tokens,
                src_key_padding_mask=valid_mask,
                triangle_pos=tri_pos,
                ray_pos=ray_token_pos,
                tf32_mode=tf32_mode,
                patch_h=patch_h,
                patch_w=patch_w,
            )
            decoded_patches = self.out_proj_act(self.out_proj(seq))
            decoded_img = rearrange(
                decoded_patches,
                "b (h1 w1) (c p1 p2) -> b c (h1 p1) (w1 p2)",
                p1=self.config.patch_size,
                p2=self.config.patch_size,
                h1=patch_h,
                w1=patch_w,
            )
            return decoded_img

    view_transformer_cls.forward = _patched_forward


class ModelVariant(StrEnum):
    """Available RenderFormer model variants."""

    V1_1_SWIN_LARGE = "v1.1_Swin_Large"


class ModelLoader(ForgeModel):
    """RenderFormer model loader implementation."""

    _VARIANTS = {
        ModelVariant.V1_1_SWIN_LARGE: ModelConfig(
            pretrained_model_name="microsoft/renderformer-v1.1-swin-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_1_SWIN_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RenderFormer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RenderFormer model from Hugging Face.

        Args:
            dtype_override: Ignored. RenderFormer uses internal NeRF positional
                encodings that produce float32 intermediates, so the model must
                remain in float32 to avoid dtype mismatches.

        Returns:
            torch.nn.Module: The RenderFormer model instance.
        """
        _ensure_renderformer_importable()
        from renderformer.models.renderformer import RenderFormer
        from renderformer.models.view_transformer import ViewTransformer
        from renderformer.layers import attention as attn_module

        _patch_view_transformer_autocast(ViewTransformer)
        _patch_swin_attn_mask_device(attn_module)

        model_name = self._variant_config.pretrained_model_name
        model = RenderFormer.from_pretrained(model_name, **kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RenderFormer model.

        Creates dummy inputs matching the RenderFormer.forward() signature.
        Inputs stay in float32 to match the model (see load_model).

        Args:
            dtype_override: Ignored to match the model's float32 requirement.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Dictionary of input tensors matching forward() parameters.
        """
        num_triangles = 128
        tex_patch_size = 1
        texture_channels = 13
        num_views = 1
        img_h = 64
        img_w = 64

        return {
            "tri_vpos_list": torch.randn(batch_size, num_triangles, 9),
            "texture_patch_list": torch.randn(
                batch_size,
                num_triangles,
                texture_channels,
                tex_patch_size,
                tex_patch_size,
            ),
            "valid_mask": torch.ones(batch_size, num_triangles, dtype=torch.bool),
            "vns": torch.randn(batch_size, num_triangles, 9),
            "rays_o": torch.randn(batch_size, num_views, 3),
            "rays_d": torch.randn(batch_size, num_views, img_h, img_w, 3),
            "tri_vpos_view_tf": torch.randn(batch_size, num_views, num_triangles, 9),
        }
