# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image model loader implementation.

Loads the diffusion transformer from Comfy-Org/LongCat-Image single-file
safetensors checkpoint. This is a Flux-based architecture without guidance
embedding support.

Available variants:
- LONGCAT_IMAGE_BF16: LongCat-Image bf16 diffusion transformer
"""

from typing import Any, Optional

import torch
from diffusers import FluxTransformer2DModel
from huggingface_hub import hf_hub_download

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

REPO_ID = "Comfy-Org/LongCat-Image"
SAFETENSORS_FILE = "split_files/diffusion_models/longcat_image_bf16.safetensors"

_TRANSFORMER_CONFIG = {
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 3584,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}

_INNER_DIM = 3072
_MLP_RATIO = 4.0


def _convert_checkpoint(checkpoint):
    """Convert BFL-format checkpoint to diffusers format.

    Handles missing vector_in (text_embedder) and uses .weight norm keys
    instead of .scale.
    """
    converted = {}

    def _swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        return torch.cat([scale, shift], dim=0)

    converted["time_text_embed.timestep_embedder.linear_1.weight"] = checkpoint.pop(
        "time_in.in_layer.weight"
    )
    converted["time_text_embed.timestep_embedder.linear_1.bias"] = checkpoint.pop(
        "time_in.in_layer.bias"
    )
    converted["time_text_embed.timestep_embedder.linear_2.weight"] = checkpoint.pop(
        "time_in.out_layer.weight"
    )
    converted["time_text_embed.timestep_embedder.linear_2.bias"] = checkpoint.pop(
        "time_in.out_layer.bias"
    )

    converted["context_embedder.weight"] = checkpoint.pop("txt_in.weight")
    converted["context_embedder.bias"] = checkpoint.pop("txt_in.bias")
    converted["x_embedder.weight"] = checkpoint.pop("img_in.weight")
    converted["x_embedder.bias"] = checkpoint.pop("img_in.bias")

    num_layers = (
        max(
            int(k.split(".", 2)[1])
            for k in checkpoint
            if k.startswith("double_blocks.")
        )
        + 1
    )
    num_single_layers = (
        max(
            int(k.split(".", 2)[1])
            for k in checkpoint
            if k.startswith("single_blocks.")
        )
        + 1
    )

    norm_suffix = "weight"

    for i in range(num_layers):
        bp = f"transformer_blocks.{i}."
        converted[f"{bp}norm1.linear.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mod.lin.weight"
        )
        converted[f"{bp}norm1.linear.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_mod.lin.bias"
        )
        converted[f"{bp}norm1_context.linear.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mod.lin.weight"
        )
        converted[f"{bp}norm1_context.linear.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mod.lin.bias"
        )

        sq, sk, sv = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0
        )
        cq, ck, cv = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
        )
        sqb, skb, svb = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
        )
        cqb, ckb, cvb = torch.chunk(
            checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
        )
        converted[f"{bp}attn.to_q.weight"] = sq
        converted[f"{bp}attn.to_q.bias"] = sqb
        converted[f"{bp}attn.to_k.weight"] = sk
        converted[f"{bp}attn.to_k.bias"] = skb
        converted[f"{bp}attn.to_v.weight"] = sv
        converted[f"{bp}attn.to_v.bias"] = svb
        converted[f"{bp}attn.add_q_proj.weight"] = cq
        converted[f"{bp}attn.add_q_proj.bias"] = cqb
        converted[f"{bp}attn.add_k_proj.weight"] = ck
        converted[f"{bp}attn.add_k_proj.bias"] = ckb
        converted[f"{bp}attn.add_v_proj.weight"] = cv
        converted[f"{bp}attn.add_v_proj.bias"] = cvb

        converted[f"{bp}attn.norm_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.{norm_suffix}"
        )
        converted[f"{bp}attn.norm_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.{norm_suffix}"
        )
        converted[f"{bp}attn.norm_added_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.{norm_suffix}"
        )
        converted[f"{bp}attn.norm_added_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.{norm_suffix}"
        )

        converted[f"{bp}ff.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mlp.0.weight"
        )
        converted[f"{bp}ff.net.0.proj.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_mlp.0.bias"
        )
        converted[f"{bp}ff.net.2.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mlp.2.weight"
        )
        converted[f"{bp}ff.net.2.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_mlp.2.bias"
        )
        converted[f"{bp}ff_context.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.weight"
        )
        converted[f"{bp}ff_context.net.0.proj.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.bias"
        )
        converted[f"{bp}ff_context.net.2.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.weight"
        )
        converted[f"{bp}ff_context.net.2.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.bias"
        )

        converted[f"{bp}attn.to_out.0.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.weight"
        )
        converted[f"{bp}attn.to_out.0.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.bias"
        )
        converted[f"{bp}attn.to_add_out.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.weight"
        )
        converted[f"{bp}attn.to_add_out.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.bias"
        )

    mlp_hidden_dim = int(_INNER_DIM * _MLP_RATIO)
    split_size = (_INNER_DIM, _INNER_DIM, _INNER_DIM, mlp_hidden_dim)

    for i in range(num_single_layers):
        bp = f"single_transformer_blocks.{i}."
        converted[f"{bp}norm.linear.weight"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.weight"
        )
        converted[f"{bp}norm.linear.bias"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.bias"
        )

        q, k, v, mlp = torch.split(
            checkpoint.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0
        )
        qb, kb, vb, mlpb = torch.split(
            checkpoint.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        )
        converted[f"{bp}attn.to_q.weight"] = q
        converted[f"{bp}attn.to_q.bias"] = qb
        converted[f"{bp}attn.to_k.weight"] = k
        converted[f"{bp}attn.to_k.bias"] = kb
        converted[f"{bp}attn.to_v.weight"] = v
        converted[f"{bp}attn.to_v.bias"] = vb
        converted[f"{bp}proj_mlp.weight"] = mlp
        converted[f"{bp}proj_mlp.bias"] = mlpb

        converted[f"{bp}attn.norm_q.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.query_norm.{norm_suffix}"
        )
        converted[f"{bp}attn.norm_k.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.key_norm.{norm_suffix}"
        )

        converted[f"{bp}proj_out.weight"] = checkpoint.pop(
            f"single_blocks.{i}.linear2.weight"
        )
        converted[f"{bp}proj_out.bias"] = checkpoint.pop(
            f"single_blocks.{i}.linear2.bias"
        )

    converted["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    converted["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")
    converted["norm_out.linear.weight"] = _swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.weight")
    )
    converted["norm_out.linear.bias"] = _swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.bias")
    )

    return converted


class ModelVariant(StrEnum):
    """Available LongCat-Image model variants."""

    LONGCAT_IMAGE_BF16 = "bf16"


class ModelLoader(ForgeModel):
    """LongCat-Image model loader for the Flux-based diffusion transformer."""

    _VARIANTS = {
        ModelVariant.LONGCAT_IMAGE_BF16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LONGCAT_IMAGE_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LONGCAT_IMAGE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> FluxTransformer2DModel:
        """Load the diffusion transformer from the single-file checkpoint.

        LongCat-Image omits the vector_in (text_embedder) and guidance_in
        layers found in standard FLUX, so we convert the checkpoint manually
        and load with strict=False.
        """
        from safetensors.torch import load_file

        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=SAFETENSORS_FILE,
        )

        model = FluxTransformer2DModel(**_TRANSFORMER_CONFIG)
        checkpoint = load_file(ckpt_path)
        converted = _convert_checkpoint(checkpoint)
        model.load_state_dict(converted, strict=False)
        model = model.to(dtype=dtype)
        model.eval()
        self._transformer = model
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the LongCat-Image diffusion transformer.

        Returns:
            FluxTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns:
            dict: Input tensors matching FluxTransformer2DModel forward signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        if self._transformer is None:
            self.load_model(dtype_override=dtype)

        config = self._transformer.config

        # Image dimensions
        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        # Prepare packed latents: (B, H*W, C)
        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Latent image IDs (seq_len, 3)
        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Text embeddings matching joint_attention_dim
        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        # Text IDs (seq_len, 3)
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        # Pooled text projections
        pooled_projections = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
