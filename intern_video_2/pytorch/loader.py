# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVideo2 Stage2 model loader implementation for video foundation tasks.

InternVideo2 is a 6B-parameter video foundation model that learns joint
video-text representations. The Stage2 checkpoint produces vision features
and CLIP-aligned embeddings that can be used for zero-shot video-text
retrieval and video classification.
"""

import sys
import types
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

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


def _patch_transformers_compat():
    """Re-add APIs removed from transformers.modeling_utils in v5.x.

    InternVideo2's remote code was written against an older transformers API.
    These helpers were moved to transformers.pytorch_utils or dropped entirely;
    we backfill them on the module object so the remote import succeeds.
    """
    import transformers.modeling_utils as mu

    if not hasattr(mu, "apply_chunking_to_forward"):
        from transformers.pytorch_utils import apply_chunking_to_forward

        mu.apply_chunking_to_forward = apply_chunking_to_forward

    if not hasattr(mu, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(
            heads, n_heads, head_size, already_pruned_heads
        ):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(mu, "prune_linear_layer"):

        def prune_linear_layer(layer, index, dim=0):
            index = index.to(layer.weight.device)
            W = layer.weight.index_select(dim, index).clone().detach()
            b = None
            if layer.bias is not None:
                b = (
                    layer.bias.clone().detach()
                    if dim == 1
                    else layer.bias[index].clone().detach()
                )
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = nn.Linear(
                new_size[1], new_size[0], bias=layer.bias is not None
            ).to(layer.weight.device)
            new_layer.weight.requires_grad = False
            new_layer.weight.copy_(W.contiguous())
            new_layer.weight.requires_grad = True
            if b is not None:
                new_layer.bias.requires_grad = False
                new_layer.bias.copy_(b.contiguous())
                new_layer.bias.requires_grad = True
            return new_layer

        mu.prune_linear_layer = prune_linear_layer


def _inject_flash_attn_stub():
    """Inject a minimal flash_attn stub so the model's module-level imports succeed.

    InternVideo2's modeling file unconditionally imports flash_attn symbols even
    when use_flash_attn=False (as in the published config.json). The stub lets
    the import pass; the actual CUDA kernels are never reached at runtime.
    """
    if "flash_attn" in sys.modules:
        return

    def _make_module(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    flash_attn = _make_module("flash_attn")
    iface = _make_module("flash_attn.flash_attn_interface")
    bert_pad = _make_module("flash_attn.bert_padding")
    mlp_mod = _make_module("flash_attn.modules")
    mlp_sub = _make_module("flash_attn.modules.mlp")
    ops_mod = _make_module("flash_attn.ops")
    rms_mod = _make_module("flash_attn.ops.rms_norm")

    flash_attn.flash_attn_interface = iface
    flash_attn.bert_padding = bert_pad
    flash_attn.modules = mlp_mod
    mlp_mod.mlp = mlp_sub
    flash_attn.ops = ops_mod
    ops_mod.rms_norm = rms_mod

    def flash_attn_varlen_qkvpacked_func(*args, **kwargs):
        raise RuntimeError(
            "flash_attn stub: should not be called when use_flash_attn=False"
        )

    iface.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func

    def unpad_input(hidden_states, attention_mask):
        raise RuntimeError(
            "flash_attn stub: should not be called when use_flash_attn=False"
        )

    def pad_input(hidden_states, indices, batch, seqlen):
        raise RuntimeError(
            "flash_attn stub: should not be called when use_flash_attn=False"
        )

    bert_pad.unpad_input = unpad_input
    bert_pad.pad_input = pad_input

    class FusedMLP(torch.nn.Module):
        pass

    mlp_sub.FusedMLP = FusedMLP

    class DropoutAddRMSNorm(torch.nn.Module):
        pass

    rms_mod.DropoutAddRMSNorm = DropoutAddRMSNorm


class ModelVariant(StrEnum):
    """Available InternVideo2 model variants."""

    STAGE2_6B = "Stage2_6B"


class ModelLoader(ForgeModel):
    """InternVideo2 Stage2 model loader for video foundation tasks."""

    _VARIANTS = {
        ModelVariant.STAGE2_6B: ModelConfig(
            pretrained_model_name="OpenGVLab/InternVideo2-Stage2_6B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STAGE2_6B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternVideo2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _inject_flash_attn_stub()
        _patch_transformers_compat()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate a synthetic video tensor matching the expected input format.

        The model's vision encoder expects input shape
        ``(batch, channels, num_frames, height, width)`` with 8 frames at
        224x224 resolution.
        """
        num_frames = 8
        image_size = 224
        channels = 3

        pixel_values = torch.randn(
            batch_size, channels, num_frames, image_size, image_size
        )

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"x": pixel_values}
