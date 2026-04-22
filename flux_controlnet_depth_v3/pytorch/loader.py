# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XLabs FLUX ControlNet Depth v3 model loader implementation.

Loads the XLabs-AI flux-controlnet-depth-v3 ControlNet from a single-file
safetensors distribution for depth-guided FLUX.1-dev image generation.

Repository: https://huggingface.co/XLabs-AI/flux-controlnet-depth-v3
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

REPO_ID = "XLabs-AI/flux-controlnet-depth-v3"
SAFETENSORS_FILE = "flux-depth-controlnet-v3.safetensors"


class ModelVariant(StrEnum):
    """Available XLabs FLUX ControlNet Depth v3 model variants."""

    DEPTH_V3 = "depth-v3"


def _convert_xlabs_controlnet_state_dict(checkpoint: dict) -> dict:
    """Convert XLabs/ComfyUI-format ControlNet keys to diffusers format."""
    converted = {}
    keys = list(checkpoint.keys())

    num_layers = (
        max(int(k.split(".")[1]) for k in keys if k.startswith("double_blocks.")) + 1
        if any(k.startswith("double_blocks.") for k in keys)
        else 0
    )

    # time_text_embed.timestep_embedder <- time_in
    converted["time_text_embed.timestep_embedder.linear_1.weight"] = checkpoint[
        "time_in.in_layer.weight"
    ]
    converted["time_text_embed.timestep_embedder.linear_1.bias"] = checkpoint[
        "time_in.in_layer.bias"
    ]
    converted["time_text_embed.timestep_embedder.linear_2.weight"] = checkpoint[
        "time_in.out_layer.weight"
    ]
    converted["time_text_embed.timestep_embedder.linear_2.bias"] = checkpoint[
        "time_in.out_layer.bias"
    ]

    # time_text_embed.text_embedder <- vector_in
    converted["time_text_embed.text_embedder.linear_1.weight"] = checkpoint[
        "vector_in.in_layer.weight"
    ]
    converted["time_text_embed.text_embedder.linear_1.bias"] = checkpoint[
        "vector_in.in_layer.bias"
    ]
    converted["time_text_embed.text_embedder.linear_2.weight"] = checkpoint[
        "vector_in.out_layer.weight"
    ]
    converted["time_text_embed.text_embedder.linear_2.bias"] = checkpoint[
        "vector_in.out_layer.bias"
    ]

    # guidance embedder <- guidance_in
    if "guidance_in.in_layer.weight" in checkpoint:
        converted["time_text_embed.guidance_embedder.linear_1.weight"] = checkpoint[
            "guidance_in.in_layer.weight"
        ]
        converted["time_text_embed.guidance_embedder.linear_1.bias"] = checkpoint[
            "guidance_in.in_layer.bias"
        ]
        converted["time_text_embed.guidance_embedder.linear_2.weight"] = checkpoint[
            "guidance_in.out_layer.weight"
        ]
        converted["time_text_embed.guidance_embedder.linear_2.bias"] = checkpoint[
            "guidance_in.out_layer.bias"
        ]

    # context_embedder <- txt_in
    converted["context_embedder.weight"] = checkpoint["txt_in.weight"]
    converted["context_embedder.bias"] = checkpoint["txt_in.bias"]

    # x_embedder <- img_in
    converted["x_embedder.weight"] = checkpoint["img_in.weight"]
    converted["x_embedder.bias"] = checkpoint["img_in.bias"]

    # controlnet_x_embedder <- pos_embed_input
    if "pos_embed_input.weight" in checkpoint:
        converted["controlnet_x_embedder.weight"] = checkpoint["pos_embed_input.weight"]
        converted["controlnet_x_embedder.bias"] = checkpoint["pos_embed_input.bias"]

    # controlnet_blocks pass-through
    for k in keys:
        if k.startswith("controlnet_blocks."):
            converted[k] = checkpoint[k]

    # double_blocks -> transformer_blocks
    for i in range(num_layers):
        bp = f"transformer_blocks.{i}."
        src = f"double_blocks.{i}."

        converted[f"{bp}norm1.linear.weight"] = checkpoint[f"{src}img_mod.lin.weight"]
        converted[f"{bp}norm1.linear.bias"] = checkpoint[f"{src}img_mod.lin.bias"]
        converted[f"{bp}norm1_context.linear.weight"] = checkpoint[
            f"{src}txt_mod.lin.weight"
        ]
        converted[f"{bp}norm1_context.linear.bias"] = checkpoint[
            f"{src}txt_mod.lin.bias"
        ]

        sample_q, sample_k, sample_v = torch.chunk(
            checkpoint[f"{src}img_attn.qkv.weight"], 3, dim=0
        )
        context_q, context_k, context_v = torch.chunk(
            checkpoint[f"{src}txt_attn.qkv.weight"], 3, dim=0
        )
        sample_q_b, sample_k_b, sample_v_b = torch.chunk(
            checkpoint[f"{src}img_attn.qkv.bias"], 3, dim=0
        )
        context_q_b, context_k_b, context_v_b = torch.chunk(
            checkpoint[f"{src}txt_attn.qkv.bias"], 3, dim=0
        )

        converted[f"{bp}attn.to_q.weight"] = sample_q
        converted[f"{bp}attn.to_q.bias"] = sample_q_b
        converted[f"{bp}attn.to_k.weight"] = sample_k
        converted[f"{bp}attn.to_k.bias"] = sample_k_b
        converted[f"{bp}attn.to_v.weight"] = sample_v
        converted[f"{bp}attn.to_v.bias"] = sample_v_b
        converted[f"{bp}attn.add_q_proj.weight"] = context_q
        converted[f"{bp}attn.add_q_proj.bias"] = context_q_b
        converted[f"{bp}attn.add_k_proj.weight"] = context_k
        converted[f"{bp}attn.add_k_proj.bias"] = context_k_b
        converted[f"{bp}attn.add_v_proj.weight"] = context_v
        converted[f"{bp}attn.add_v_proj.bias"] = context_v_b

        converted[f"{bp}attn.norm_q.weight"] = checkpoint[
            f"{src}img_attn.norm.query_norm.scale"
        ]
        converted[f"{bp}attn.norm_k.weight"] = checkpoint[
            f"{src}img_attn.norm.key_norm.scale"
        ]
        converted[f"{bp}attn.norm_added_q.weight"] = checkpoint[
            f"{src}txt_attn.norm.query_norm.scale"
        ]
        converted[f"{bp}attn.norm_added_k.weight"] = checkpoint[
            f"{src}txt_attn.norm.key_norm.scale"
        ]

        converted[f"{bp}attn.to_out.0.weight"] = checkpoint[
            f"{src}img_attn.proj.weight"
        ]
        converted[f"{bp}attn.to_out.0.bias"] = checkpoint[f"{src}img_attn.proj.bias"]
        converted[f"{bp}attn.to_add_out.weight"] = checkpoint[
            f"{src}txt_attn.proj.weight"
        ]
        converted[f"{bp}attn.to_add_out.bias"] = checkpoint[f"{src}txt_attn.proj.bias"]

        converted[f"{bp}ff.net.0.proj.weight"] = checkpoint[f"{src}img_mlp.0.weight"]
        converted[f"{bp}ff.net.0.proj.bias"] = checkpoint[f"{src}img_mlp.0.bias"]
        converted[f"{bp}ff.net.2.weight"] = checkpoint[f"{src}img_mlp.2.weight"]
        converted[f"{bp}ff.net.2.bias"] = checkpoint[f"{src}img_mlp.2.bias"]
        converted[f"{bp}ff_context.net.0.proj.weight"] = checkpoint[
            f"{src}txt_mlp.0.weight"
        ]
        converted[f"{bp}ff_context.net.0.proj.bias"] = checkpoint[
            f"{src}txt_mlp.0.bias"
        ]
        converted[f"{bp}ff_context.net.2.weight"] = checkpoint[f"{src}txt_mlp.2.weight"]
        converted[f"{bp}ff_context.net.2.bias"] = checkpoint[f"{src}txt_mlp.2.bias"]

    return converted


class ModelLoader(ForgeModel):
    """XLabs FLUX ControlNet Depth v3 model loader for depth-guided image generation."""

    _VARIANTS = {
        ModelVariant.DEPTH_V3: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEPTH_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="XLabs FLUX ControlNet Depth v3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the XLabs FLUX ControlNet Depth v3 model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            FluxControlNetModel: The ControlNet model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        safetensors_path = hf_hub_download(REPO_ID, SAFETENSORS_FILE)
        checkpoint = load_file(safetensors_path)

        num_layers = (
            max(
                int(k.split(".")[1])
                for k in checkpoint
                if k.startswith("double_blocks.")
            )
            + 1
            if any(k.startswith("double_blocks.") for k in checkpoint)
            else 0
        )
        has_guidance = "guidance_in.in_layer.weight" in checkpoint

        self.controlnet = FluxControlNetModel(
            num_layers=num_layers,
            num_single_layers=0,
            guidance_embeds=has_guidance,
        )

        converted = _convert_xlabs_controlnet_state_dict(checkpoint)
        self.controlnet.load_state_dict(converted, strict=False)
        self.controlnet = self.controlnet.to(dtype=compute_dtype)
        self.controlnet.eval()
        return self.controlnet

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, batch_size: int = 1
    ) -> Any:
        """Prepare sample inputs for the ControlNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size (default: 1).

        Returns:
            dict: Input tensors matching FluxControlNetModel.forward() signature.
        """
        if self.controlnet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.controlnet.config

        in_channels = config.in_channels
        joint_attention_dim = config.joint_attention_dim
        pooled_projection_dim = config.pooled_projection_dim

        img_seq_len = 64
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
        guidance = (
            torch.full([batch_size], 3.5, dtype=dtype)
            if config.guidance_embeds
            else None
        )

        inputs = {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
        if guidance is not None:
            inputs["guidance"] = guidance
        return inputs
