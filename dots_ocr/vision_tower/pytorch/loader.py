# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr vision-tower loader - the DotsVisionTransformer image encoder alone.

This isolates the NaViT-style vision tower (model_type ``dots_vit``: 42 blocks,
patch 14, RMSNorm, SwiGLU FFN, 2D vision-rotary attention, spatial-merge=2
PatchMerger) so it can be validated on device independently of the Qwen2
decoder.

Device note: the stock tower derives rotary position ids / ``cu_seqlens`` from
the patch grid with ``torch.arange``/``.max()``/python loops over per-image h/w.
Those run against device-resident scalars and fail to compile (``torch.arange``
of a device scalar). They are pure functions of the *static* image grid, so we
precompute the rotary table on the host (``compute_vision_rotary``) and feed it
in, and - because a single image attends fully over all its own patches
(``cu_seqlens = [0, N]`` ⇒ all-zero additive mask) - run plain full attention on
device. The heavy patch-embed + 42 transformer blocks + merger run on device.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...common import (
    DOTS_OCR_MODEL,
    load_full_model,
    load_processor,
    build_multimodal_inputs,
    compute_vision_rotary,
)


def _apply_rotary(tensor, freqs):
    """Replica of modeling_dots_vision.apply_rotary_pos_emb_vision."""
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    x1 = tensor[..., : tensor.shape[-1] // 2]
    x2 = tensor[..., tensor.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((tensor * cos) + (rotated * sin)).to(orig_dtype)


class _VisionTowerWrapper(nn.Module):
    """DotsVisionTransformer with host-precomputed rotary + full attention.

    forward(pixel_values, rotary_pos_emb) -> merged image embeddings
    [num_patches // merge^2, hidden_size]. Equivalent to the stock tower for a
    single image, but with all grid-derived host control flow lifted out.
    """

    def __init__(self, vision_tower):
        super().__init__()
        self.vt = vision_tower
        self.num_heads = vision_tower.config.num_attention_heads
        self.head_dim = vision_tower.config.embed_dim // self.num_heads

    def _patch_embed(self, pixel_values):
        """Patch embedding as a matmul instead of Conv2d.

        The stock patch embed is ``Conv2d(3, embed_dim, kernel=stride=patch)``,
        i.e. a per-patch linear projection - but ttnn's conv2d fails at runtime
        on the kernel==stride==full-input configuration. Since pixel_values are
        already flattened patches [num_patches, C*patch*patch] in the same
        (c, h, w) order as the conv weight, this is exactly a Linear with the
        conv weight reshaped to [embed_dim, C*patch*patch].
        """
        pe = self.vt.patch_embed.patchifier
        conv = pe.proj
        w = conv.weight.reshape(conv.weight.shape[0], -1)  # [embed_dim, C*p*p]
        x = F.linear(pixel_values, w, conv.bias)  # [num_patches, embed_dim]
        return pe.norm(x)

    def _attn(self, attn, hidden, rotary):
        seq = hidden.shape[0]
        q, k, v = (
            attn.qkv(hidden).reshape(seq, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        q = _apply_rotary(q.unsqueeze(0), rotary).squeeze(0)
        k = _apply_rotary(k.unsqueeze(0), rotary).squeeze(0)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_w = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        # Single image: full bidirectional attention over all patches (no mask).
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v).transpose(0, 1).reshape(seq, -1)
        return attn.proj(out)

    def forward(self, pixel_values, rotary_pos_emb):
        vt = self.vt
        pixel_values = pixel_values.to(vt.dtype)
        rotary_pos_emb = rotary_pos_emb.to(torch.float32)
        h = self._patch_embed(pixel_values)
        for blk in vt.blocks:
            h = h + self._attn(blk.attn, blk.norm1(h), rotary_pos_emb)
            h = h + blk.mlp(blk.norm2(h))
        if vt.config.post_norm:
            h = vt.post_trunk_norm(h)
        h = vt.merger(h)
        return h


class ModelVariant(StrEnum):
    """Available dots.ocr vision-tower variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Loader for the dots.ocr vision tower (DotsVisionTransformer)."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name=DOTS_OCR_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._spatial_merge_size = 2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="dots_ocr_vision_tower",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        if self.processor is None:
            self.processor = load_processor()
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full model and return its vision tower wrapped for forward."""
        model = load_full_model(dtype_override=dtype_override)
        vt = model.vision_tower
        self._spatial_merge_size = vt.config.spatial_merge_size
        wrapper = _VisionTowerWrapper(vt)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return (pixel_values, rotary_pos_emb) for the vision tower.

        rotary_pos_emb is precomputed on host from the patch grid so the device
        graph carries no grid-derived control flow.
        """
        self._load_processor()
        full = build_multimodal_inputs(self.processor, dtype_override=dtype_override)
        grid_thw = full["image_grid_thw"]
        head_dim = 128  # embed_dim 1536 / 12 heads
        rotary = compute_vision_rotary(
            grid_thw, head_dim=head_dim, spatial_merge_size=self._spatial_merge_size
        )
        return {
            "pixel_values": full["pixel_values"],
            "rotary_pos_emb": rotary,
        }
