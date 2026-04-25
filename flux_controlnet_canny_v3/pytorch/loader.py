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

from typing import Any, Dict, Optional

import torch
from diffusers import FluxControlNetModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

REPO_ID = "XLabs-AI/flux-controlnet-canny-v3"
CHECKPOINT_FILE = "flux-canny-controlnet-v3.safetensors"

# XLabs ControlNets use a much smaller architecture than the InstantX union
# ControlNet: 2 double-stream transformer blocks and no single-stream blocks.
_NUM_LAYERS = 2
_NUM_SINGLE_LAYERS = 0


def _convert_bfl_to_diffusers(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Convert BFL/XLabs checkpoint keys to diffusers FluxControlNetModel keys.

    The XLabs checkpoint uses BFL-style naming (double_blocks.*, img_in.*, etc.)
    while diffusers expects transformer_blocks.*, x_embedder.*, etc.
    QKV tensors are also split from combined (3*d, d) into separate Q/K/V matrices.

    Keys without a diffusers equivalent (input_hint_block.*, guidance_in.*) are
    intentionally omitted; those weights have no counterpart in FluxControlNetModel.
    """
    new_sd: Dict[str, torch.Tensor] = {}

    simple_renames = {
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
    }
    for old, new in simple_renames.items():
        if old in state_dict:
            new_sd[new] = state_dict[old]

    block_indices = sorted(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("double_blocks.")}
    )
    for i in block_indices:
        src = f"double_blocks.{i}"
        dst = f"transformer_blocks.{i}"

        # Modulation / norm layers
        for suffix in ("weight", "bias"):
            if f"{src}.img_mod.lin.{suffix}" in state_dict:
                new_sd[f"{dst}.norm1.linear.{suffix}"] = state_dict[
                    f"{src}.img_mod.lin.{suffix}"
                ]
            if f"{src}.txt_mod.lin.{suffix}" in state_dict:
                new_sd[f"{dst}.norm1_context.linear.{suffix}"] = state_dict[
                    f"{src}.txt_mod.lin.{suffix}"
                ]

        # Image attention: split combined QKV → Q, K, V
        for param in ("weight", "bias"):
            key = f"{src}.img_attn.qkv.{param}"
            if key in state_dict:
                qkv = state_dict[key]
                d = qkv.shape[0] // 3
                new_sd[f"{dst}.attn.to_q.{param}"] = qkv[:d]
                new_sd[f"{dst}.attn.to_k.{param}"] = qkv[d : 2 * d]
                new_sd[f"{dst}.attn.to_v.{param}"] = qkv[2 * d :]

        # Image attention RMSNorm scale → weight
        for src_key, dst_key in (
            (f"{src}.img_attn.norm.query_norm.scale", f"{dst}.attn.norm_q.weight"),
            (f"{src}.img_attn.norm.key_norm.scale", f"{dst}.attn.norm_k.weight"),
        ):
            if src_key in state_dict:
                new_sd[dst_key] = state_dict[src_key]

        # Image attention output projection
        for suffix in ("weight", "bias"):
            if f"{src}.img_attn.proj.{suffix}" in state_dict:
                new_sd[f"{dst}.attn.to_out.0.{suffix}"] = state_dict[
                    f"{src}.img_attn.proj.{suffix}"
                ]

        # Text attention: split combined QKV → add_q_proj, add_k_proj, add_v_proj
        for param in ("weight", "bias"):
            key = f"{src}.txt_attn.qkv.{param}"
            if key in state_dict:
                qkv = state_dict[key]
                d = qkv.shape[0] // 3
                new_sd[f"{dst}.attn.add_q_proj.{param}"] = qkv[:d]
                new_sd[f"{dst}.attn.add_k_proj.{param}"] = qkv[d : 2 * d]
                new_sd[f"{dst}.attn.add_v_proj.{param}"] = qkv[2 * d :]

        # Text attention RMSNorm scale → weight
        for src_key, dst_key in (
            (
                f"{src}.txt_attn.norm.query_norm.scale",
                f"{dst}.attn.norm_added_q.weight",
            ),
            (f"{src}.txt_attn.norm.key_norm.scale", f"{dst}.attn.norm_added_k.weight"),
        ):
            if src_key in state_dict:
                new_sd[dst_key] = state_dict[src_key]

        # Text attention output projection
        for suffix in ("weight", "bias"):
            if f"{src}.txt_attn.proj.{suffix}" in state_dict:
                new_sd[f"{dst}.attn.to_add_out.{suffix}"] = state_dict[
                    f"{src}.txt_attn.proj.{suffix}"
                ]

        # Image MLP
        for suffix in ("weight", "bias"):
            if f"{src}.img_mlp.0.{suffix}" in state_dict:
                new_sd[f"{dst}.ff.net.0.proj.{suffix}"] = state_dict[
                    f"{src}.img_mlp.0.{suffix}"
                ]
            if f"{src}.img_mlp.2.{suffix}" in state_dict:
                new_sd[f"{dst}.ff.net.2.{suffix}"] = state_dict[
                    f"{src}.img_mlp.2.{suffix}"
                ]

        # Text MLP
        for suffix in ("weight", "bias"):
            if f"{src}.txt_mlp.0.{suffix}" in state_dict:
                new_sd[f"{dst}.ff_context.net.0.proj.{suffix}"] = state_dict[
                    f"{src}.txt_mlp.0.{suffix}"
                ]
            if f"{src}.txt_mlp.2.{suffix}" in state_dict:
                new_sd[f"{dst}.ff_context.net.2.{suffix}"] = state_dict[
                    f"{src}.txt_mlp.2.{suffix}"
                ]

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
        )

        raw_sd = load_file(model_path)
        converted_sd = _convert_bfl_to_diffusers(raw_sd)
        self._controlnet.load_state_dict(converted_sd, strict=False)
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

        return {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
