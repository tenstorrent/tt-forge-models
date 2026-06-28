# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr vision tower (NaViT image encoder) loader implementation.

dots.ocr's image encoder is a custom NaViT-style ViT (``DotsVisionTransformer``):
a Conv2d patch embed (kernel=stride=14), RMSNorm pre-norm blocks with 2D vision
RoPE, SwiGLU FFN, and a 2x2 spatial PatchMerger. The reference forward computes
``cu_seqlens`` and the rotary position embedding from ``grid_thw`` *inside* the graph
(Python loops over tensor rows, ``.max()``/``.item()``, data-dependent slicing to build
the per-image attention mask). Those are data-dependent ops that the static-shape device
path cannot compile.

This loader brings the encoder up as a clean static graph by wrapping it: the rotary
position embedding is precomputed on the host (it is a pure function of ``grid_thw``) and
passed in as a tensor, and attention runs as dense full attention over the single image
(for one image ``cu_seqlens = [0, S]`` makes the reference mask all-zeros, i.e. full
attention — so no masking is required). The wrapper reuses the real pretrained weights.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Qwen2VLImageProcessor
from typing import Optional
from PIL import Image

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"

# Image side length (pixels). 224 is divisible by patch_size*merge_size (28), giving a
# 16x16 patch grid => 256 patch tokens, kept small for a fast first compile.
_IMG_SIZE = 224


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """2D vision RoPE — matches modeling_dots_vision.apply_rotary_pos_emb_vision."""
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


class DotsVisionStaticWrapper(nn.Module):
    """Static-graph wrapper around DotsVisionTransformer for single-image encoding.

    Reuses the pretrained submodules (transformer blocks, post-trunk norm, patch merger)
    but takes the precomputed rotary position embedding as an input and runs dense full
    attention, avoiding the reference forward's data-dependent control flow
    (``cu_seqlens`` construction, ``.max()``/``.item()``, Python mask loop).

    The patch embed's ``nn.Conv2d`` (kernel=stride=patch_size, producing a 1x1 output per
    patch) is replaced by its mathematically-equivalent linear projection over flattened
    patches: ttnn's conv2d program factory rejects this degenerate conv at runtime
    (``Reader indices buffer page size N exceeds worst-case CB size M``), while the
    flatten+matmul form maps to a standard supported matmul.
    """

    def __init__(self, vision_tower):
        super().__init__()
        patchifier = vision_tower.patch_embed.patchifier
        conv = patchifier.proj  # nn.Conv2d(num_channels, embed_dim, kernel=stride=patch)
        embed_dim = conv.weight.shape[0]
        # Conv2d with kernel=stride and 1x1 output == Linear over the flattened patch.
        # pixel_values arrive pre-flattened as [S, num_channels*patch*patch] in the same
        # (channel, ph, pw) order as conv.weight.reshape(embed_dim, -1).
        self.patch_weight = nn.Parameter(
            conv.weight.reshape(embed_dim, -1).clone(), requires_grad=False
        )
        self.patch_bias = (
            nn.Parameter(conv.bias.clone(), requires_grad=False)
            if conv.bias is not None
            else None
        )
        self.patch_norm = patchifier.norm  # RMSNorm over embed_dim

        self.blocks = vision_tower.blocks
        self.post_norm = getattr(vision_tower, "post_trunk_norm", None)
        self.merger = vision_tower.merger
        self.num_heads = vision_tower.config.num_attention_heads
        self.head_dim = embed_dim // self.num_heads

    def _patch_embed(self, pixel_values):
        x = nn.functional.linear(pixel_values, self.patch_weight, self.patch_bias)
        return self.patch_norm(x)

    def _attn(self, attn_mod, hidden_states, rotary_pos_emb):
        seq_length = hidden_states.shape[0]
        q, k, v = (
            attn_mod.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # Single image => full attention (reference mask is all-zeros).
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
        return attn_mod.proj(attn_output)

    def forward(self, pixel_values, rotary_pos_emb):
        hidden_states = self._patch_embed(pixel_values)
        for blk in self.blocks:
            hidden_states = hidden_states + self._attn(
                blk.attn, blk.norm1(hidden_states), rotary_pos_emb
            )
            hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))
        if self.post_norm is not None:
            hidden_states = self.post_norm(hidden_states)
        hidden_states = self.merger(hidden_states)
        return hidden_states


class ModelVariant(StrEnum):
    """Available dots.ocr vision-tower variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr vision tower (NaViT image encoder) loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.image_processor = None
        self.grid_thw = None
        self.dtype = None
        # Bound DotsVisionTransformer.rot_pos_emb, captured in load_model so the rotary
        # position embedding is computed by the model's own (exact) host-side routine.
        self._rot_pos_emb = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return model metadata for the given variant."""
        return ModelInfo(
            model="dots.ocr vision tower",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_image(self):
        """Deterministic synthetic RGB document-like image (no network dependency)."""
        rng = np.random.default_rng(0)
        # White background with scattered dark rectangles to mimic text blocks.
        arr = np.full((_IMG_SIZE, _IMG_SIZE, 3), 245, dtype=np.uint8)
        for _ in range(40):
            y = int(rng.integers(0, _IMG_SIZE - 12))
            x = int(rng.integers(0, _IMG_SIZE - 30))
            arr[y : y + 6, x : x + 24] = int(rng.integers(0, 80))
        return Image.fromarray(arr, mode="RGB")

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the dots.ocr vision tower wrapped for static-graph compilation.

        Args:
            dtype_override: Optional torch dtype to load weights in (e.g. bfloat16).

        Returns:
            torch.nn.Module: ``DotsVisionStaticWrapper`` over the pretrained vision tower.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Default to float32 on CPU; the runner passes dtype_override=bfloat16 for device.
        self.dtype = dtype_override if dtype_override is not None else torch.float32
        model_kwargs = {
            "trust_remote_code": True,
            "revision": _REVISION,
            "dtype": self.dtype,
        }

        full_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs, **kwargs
        )
        vision_tower = full_model.vision_tower
        # Drop the language model so only vision-tower weights are resident.
        if hasattr(full_model, "model"):
            full_model.model = None
        if hasattr(full_model, "lm_head"):
            full_model.lm_head = None

        # transformers' weight-loading path leaves VisionRotaryEmbedding's non-persistent
        # ``inv_freq`` buffer (computed in __init__, persistent=False so absent from the
        # state dict) uninitialized/garbage — observed abs values up to ~1e9, which makes
        # the 2D vision RoPE numerically meaningless. Recompute it (theta=10000) so the
        # rotary frequency table is correct and stable across CPU/device.
        rot = vision_tower.rotary_pos_emb
        rope_dim = rot.inv_freq.numel() * 2
        rot.inv_freq = (
            1.0
            / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))
        ).to(rot.inv_freq.dtype)

        # Capture the model's own rotary routine (keeps vision_tower alive) so the
        # precomputed rotary_pos_emb passed at load_inputs is bit-exact with the model.
        self._rot_pos_emb = vision_tower.rot_pos_emb

        model = DotsVisionStaticWrapper(vision_tower)
        model = model.to(self.dtype)
        model.eval()
        self.model = model
        return model

    def _load_image_processor(self):
        self.image_processor = Qwen2VLImageProcessor(
            patch_size=14,
            temporal_patch_size=1,
            merge_size=2,
            min_pixels=56 * 56,
            max_pixels=_IMG_SIZE * _IMG_SIZE,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        return self.image_processor

    def load_inputs(self, dtype_override=None):
        """Return ``{"pixel_values", "rotary_pos_emb"}`` for the vision tower wrapper.

        ``pixel_values`` is the flattened-patch tensor [S, 588] from the Qwen2VL image
        processor; ``rotary_pos_emb`` is the host-precomputed RoPE frequency table.

        Args:
            dtype_override: Optional torch dtype for ``pixel_values`` (e.g. bfloat16).

        Returns:
            dict: Inputs for ``DotsVisionStaticWrapper.forward``.
        """
        if self.image_processor is None:
            self._load_image_processor()
        if self._rot_pos_emb is None:
            raise RuntimeError("load_model must be called before load_inputs.")

        image = self._make_image()
        processed = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        grid_thw = processed["image_grid_thw"]  # [[t, h, w]]
        self.grid_thw = grid_thw

        # Use the model's own rotary computation for a bit-exact frequency table.
        rotary_pos_emb = self._rot_pos_emb(grid_thw)

        # Match pixel_values to the model's weight dtype (set in load_model).
        target_dtype = dtype_override or self.dtype or torch.float32
        pixel_values = pixel_values.to(target_dtype)

        return {"pixel_values": pixel_values, "rotary_pos_emb": rotary_pos_emb}
