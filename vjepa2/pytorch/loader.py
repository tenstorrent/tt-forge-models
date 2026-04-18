# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V-JEPA2 model loader implementation for video classification.
"""

import types
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor

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


class ModelVariant(StrEnum):
    """Available V-JEPA2 model variants."""

    VITL_FPC64_256 = "vitl_fpc64_256"


def _eager_attention_forward(
    self, hidden_states, position_mask=None, output_attentions=False
):
    batch_size, seq_length, _ = hidden_states.shape
    query_layer = (
        self.query(hidden_states)
        .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
    )
    key_layer = (
        self.key(hidden_states)
        .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
    )
    value_layer = (
        self.value(hidden_states)
        .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
    )

    pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
    key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
    query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

    attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scaling
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_layer.dtype
    )
    attn_output = torch.matmul(attn_weights, value_layer)
    attn_output = attn_output.transpose(1, 2).contiguous()

    new_shape = attn_output.size()[:-2] + (self.all_head_size,)
    context_layer = self.proj(attn_output.reshape(new_shape))

    outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
    return outputs


class _SimpleVJEPA2Layer(torch.nn.Module):
    """Plain nn.Module layer that bypasses GradientCheckpointingLayer.__call__."""

    def __init__(self, layer):
        super().__init__()
        self.norm1 = layer.norm1
        self.attention = layer.attention
        self.norm2 = layer.norm2
        self.mlp = layer.mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attention_output = self.attention(hidden_states)[0]
        hidden_states = attention_output + residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


def _patch_tt_backend_strip_guards():
    """Patch the tt backend to strip _guards_fn nodes before torch.export.

    Dynamo inserts _guards_fn submodules referencing local variable dicts
    that become undefined during re-export inside the tt backend pipeline.
    Since these guards have num_users=0, removing them is safe.
    """
    try:
        import tt_torch.backend.backend as backend
    except ImportError:
        return

    original = backend.torch_pass_pipeline
    if getattr(original, "_guards_patched", False):
        return

    def patched_pipeline(gm, example_inputs, options):
        for node in list(gm.graph.nodes):
            if "_guards" in str(node.target):
                gm.graph.erase_node(node)
        gm.recompile()
        for name in [n for n in dir(gm) if "_guards" in n]:
            try:
                delattr(gm, name)
            except AttributeError:
                pass
        return original(gm, example_inputs, options)

    patched_pipeline._guards_patched = True
    backend.torch_pass_pipeline = patched_pipeline


class VJEPA2EncoderWrapper(torch.nn.Module):
    """Wraps VJEPA2 encoder, patching attention to avoid dynamo guard failures."""

    def __init__(self, model):
        super().__init__()
        encoder = model.encoder
        self.patch_embeddings = encoder.embeddings.patch_embeddings
        self.layernorm = encoder.layernorm

        self.layers = nn.ModuleList()
        for layer in encoder.layer:
            layer.attention.forward = types.MethodType(
                _eager_attention_forward, layer.attention
            )
            self.layers.append(_SimpleVJEPA2Layer(layer))

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)
        pixel_values_videos = pixel_values_videos.to(
            dtype=self.patch_embeddings.proj.weight.dtype
        )
        hidden_states = self.patch_embeddings(pixel_values_videos)
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return self.layernorm(hidden_states)


class ModelLoader(ForgeModel):
    """V-JEPA2 model loader for video classification."""

    _VARIANTS = {
        ModelVariant.VITL_FPC64_256: ModelConfig(
            pretrained_model_name="facebook/vjepa2-vitl-fpc64-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITL_FPC64_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize V-JEPA2 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="V-JEPA2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the V-JEPA2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModel.from_pretrained(
            model_name, attn_implementation="eager", **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        self.processor = AutoVideoProcessor.from_pretrained(model_name)

        _patch_tt_backend_strip_guards()
        return VJEPA2EncoderWrapper(model)

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load and return input tensors for V-JEPA2."""
        if self.processor is None:
            raise RuntimeError(
                "Model must be loaded first before loading inputs. Call load_model() first."
            )

        # Create synthetic video: 64 frames of 256x256 RGB
        video = np.random.randint(0, 255, (64, 256, 256, 3), dtype=np.uint8)

        inputs = self.processor(video, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: (
                    v.to(dtype_override)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v
                )
                for k, v in inputs.items()
            }

        return dict(inputs)
