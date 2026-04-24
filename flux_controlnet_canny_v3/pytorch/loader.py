# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX ControlNet Canny v3 model loader implementation.

Loads the XLabs-AI/flux-controlnet-canny-v3 ControlNet from the single-file
safetensors checkpoint shipped in the repo (no config.json is provided).

Available variants:
- XLABS_V3: XLabs-AI canny edge ControlNet v3 for FLUX.1-dev
"""

from typing import Any, Optional

import torch
from diffusers import FluxControlNetModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

REPO_ID = "XLabs-AI/flux-controlnet-canny-v3"
CHECKPOINT_FILE = "flux-canny-controlnet-v3.safetensors"

# XLabs ControlNets use 2 double-stream transformer blocks and no single-stream blocks.
_NUM_LAYERS = 2
_NUM_SINGLE_LAYERS = 0


def _remap_xlabs_to_diffusers(state_dict):
    """Remap XLabs-style checkpoint keys to diffusers FluxControlNetModel keys.

    XLabs checkpoints use a different naming convention than diffusers. The
    input_hint_block (pixel-space CNN) is skipped as it has no diffusers
    equivalent; pos_embed_input maps to controlnet_x_embedder. QKV weights
    are split from concatenated form into separate Q/K/V tensors.
    """
    new_sd = {}
    for key, value in state_dict.items():
        # Skip pixel-space conditioning CNN; no equivalent in diffusers model
        if key.startswith("input_hint_block"):
            continue

        # Top-level renames
        simple_map = {
            "img_in.weight": "x_embedder.weight",
            "img_in.bias": "x_embedder.bias",
            "txt_in.weight": "context_embedder.weight",
            "txt_in.bias": "context_embedder.bias",
            "pos_embed_input.weight": "controlnet_x_embedder.weight",
            "pos_embed_input.bias": "controlnet_x_embedder.bias",
            "time_in.in_layer.weight": "time_text_embed.timestep_embedder.linear_1.weight",
            "time_in.in_layer.bias": "time_text_embed.timestep_embedder.linear_1.bias",
            "time_in.out_layer.weight": "time_text_embed.timestep_embedder.linear_2.weight",
            "time_in.out_layer.bias": "time_text_embed.timestep_embedder.linear_2.bias",
            "vector_in.in_layer.weight": "time_text_embed.text_embedder.linear_1.weight",
            "vector_in.in_layer.bias": "time_text_embed.text_embedder.linear_1.bias",
            "vector_in.out_layer.weight": "time_text_embed.text_embedder.linear_2.weight",
            "vector_in.out_layer.bias": "time_text_embed.text_embedder.linear_2.bias",
            "guidance_in.in_layer.weight": "time_text_embed.guidance_embedder.linear_1.weight",
            "guidance_in.in_layer.bias": "time_text_embed.guidance_embedder.linear_1.bias",
            "guidance_in.out_layer.weight": "time_text_embed.guidance_embedder.linear_2.weight",
            "guidance_in.out_layer.bias": "time_text_embed.guidance_embedder.linear_2.bias",
        }
        if key in simple_map:
            new_sd[simple_map[key]] = value
            continue

        if not key.startswith("double_blocks."):
            continue

        parts = key.split(".", 2)
        block_idx = parts[1]
        rest = parts[2]
        pfx = f"transformer_blocks.{block_idx}."

        double_block_map = {
            "img_mod.lin.weight": pfx + "norm1.linear.weight",
            "img_mod.lin.bias": pfx + "norm1.linear.bias",
            "txt_mod.lin.weight": pfx + "norm1_context.linear.weight",
            "txt_mod.lin.bias": pfx + "norm1_context.linear.bias",
            "img_attn.norm.query_norm.scale": pfx + "attn.norm_q.weight",
            "img_attn.norm.key_norm.scale": pfx + "attn.norm_k.weight",
            "img_attn.proj.weight": pfx + "attn.to_out.0.weight",
            "img_attn.proj.bias": pfx + "attn.to_out.0.bias",
            "txt_attn.norm.query_norm.scale": pfx + "attn.norm_added_q.weight",
            "txt_attn.norm.key_norm.scale": pfx + "attn.norm_added_k.weight",
            "txt_attn.proj.weight": pfx + "attn.to_add_out.weight",
            "txt_attn.proj.bias": pfx + "attn.to_add_out.bias",
            "img_mlp.0.weight": pfx + "ff.net.0.proj.weight",
            "img_mlp.0.bias": pfx + "ff.net.0.proj.bias",
            "img_mlp.2.weight": pfx + "ff.net.2.weight",
            "img_mlp.2.bias": pfx + "ff.net.2.bias",
            "txt_mlp.0.weight": pfx + "ff_context.net.0.proj.weight",
            "txt_mlp.0.bias": pfx + "ff_context.net.0.proj.bias",
            "txt_mlp.2.weight": pfx + "ff_context.net.2.weight",
            "txt_mlp.2.bias": pfx + "ff_context.net.2.bias",
        }
        if rest in double_block_map:
            new_sd[double_block_map[rest]] = value
            continue

        # Split concatenated QKV → separate Q / K / V
        if rest in ("img_attn.qkv.weight", "img_attn.qkv.bias"):
            d = value.shape[0] // 3
            is_weight = rest.endswith(".weight")
            new_sd[pfx + "attn.to_q." + ("weight" if is_weight else "bias")] = value[:d]
            new_sd[pfx + "attn.to_k." + ("weight" if is_weight else "bias")] = value[
                d : 2 * d
            ]
            new_sd[pfx + "attn.to_v." + ("weight" if is_weight else "bias")] = value[
                2 * d :
            ]
        elif rest in ("txt_attn.qkv.weight", "txt_attn.qkv.bias"):
            d = value.shape[0] // 3
            is_weight = rest.endswith(".weight")
            new_sd[
                pfx + "attn.add_q_proj." + ("weight" if is_weight else "bias")
            ] = value[:d]
            new_sd[
                pfx + "attn.add_k_proj." + ("weight" if is_weight else "bias")
            ] = value[d : 2 * d]
            new_sd[
                pfx + "attn.add_v_proj." + ("weight" if is_weight else "bias")
            ] = value[2 * d :]

    return new_sd


class ModelVariant(StrEnum):
    """Available FLUX ControlNet Canny v3 variants."""

    XLABS_V3 = "XLabs-v3"


class ModelLoader(ForgeModel):
    """FLUX ControlNet Canny v3 model loader."""

    _VARIANTS = {
        ModelVariant.XLABS_V3: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.XLABS_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_CONTROLNET_CANNY_V3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_controlnet(
        self, dtype: torch.dtype = torch.float32
    ) -> FluxControlNetModel:
        """Load the ControlNet from the single-file safetensors checkpoint."""
        model_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE)

        self._controlnet = FluxControlNetModel(
            num_layers=_NUM_LAYERS,
            num_single_layers=_NUM_SINGLE_LAYERS,
            guidance_embeds=True,
        )

        state_dict = load_file(model_path)
        remapped = _remap_xlabs_to_diffusers(state_dict)
        self._controlnet.load_state_dict(remapped, strict=False)
        self._controlnet.to(dtype=dtype)
        self._controlnet.eval()
        return self._controlnet

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the ControlNet model.

        Returns:
            FluxControlNetModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._controlnet is None:
            return self._load_controlnet(dtype)
        if dtype_override is not None:
            self._controlnet = self._controlnet.to(dtype=dtype_override)
        return self._controlnet

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Prepare sample inputs for the ControlNet.

        Returns a dict matching FluxControlNetModel.forward() signature.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        # FluxControlNetModel default config values
        in_channels = 64
        joint_attention_dim = 4096
        pooled_projection_dim = 768

        # Sequence lengths for image and text tokens
        img_seq_len = 64  # e.g. 8x8 patch grid
        txt_seq_len = 32

        hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        controlnet_cond = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_ids = torch.zeros(img_seq_len, 3, dtype=dtype)
        txt_ids = torch.zeros(txt_seq_len, 3, dtype=dtype)
        # guidance_embeds=True requires a guidance scale tensor
        guidance = torch.tensor([3.5] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,
        }
