# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dark Beast FLUX.2 Klein LoRA model loader implementation.

Loads the FLUX.2 Klein base pipeline and applies Dark Beast LoRA weights
from Keltezaa/Dark_Beast for dark, beast-themed text-to-image generation.

Available variants:
- DARK_BEAST: Dark Beast LoRA applied to FLUX.2 Klein
"""

from typing import Optional

import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import hf_hub_download
from safetensors import safe_open

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

BASE_MODEL = "Runware/BFL-FLUX.2-klein-9B"
LORA_REPO = "Keltezaa/Dark_Beast"
LORA_WEIGHT_NAME = "darkBeastMar0326Latest_dbkleinv2BFS.safetensors"


def _convert_comfyui_to_diffusers(state_dict):
    """Convert ComfyUI-format FLUX.2 checkpoint keys to HuggingFace diffusers format.

    The Dark Beast checkpoint uses ComfyUI naming (model.diffusion_model.*) with
    fused QKV weights, while diffusers uses separate Q/K/V projections.
    """
    new_sd = {}
    for orig_key, tensor in state_dict.items():
        key = orig_key.replace("model.diffusion_model.", "")

        if key.startswith("double_blocks."):
            parts = key.split(".")
            idx = parts[1]
            rest = ".".join(parts[2:])
            if rest == "img_attn.qkv.weight":
                q, k, v = tensor.chunk(3, dim=0)
                new_sd[f"transformer_blocks.{idx}.attn.to_q.weight"] = q
                new_sd[f"transformer_blocks.{idx}.attn.to_k.weight"] = k
                new_sd[f"transformer_blocks.{idx}.attn.to_v.weight"] = v
            elif rest == "img_attn.proj.weight":
                new_sd[f"transformer_blocks.{idx}.attn.to_out.0.weight"] = tensor
            elif rest == "img_attn.norm.query_norm.scale":
                new_sd[f"transformer_blocks.{idx}.attn.norm_q.weight"] = tensor
            elif rest == "img_attn.norm.key_norm.scale":
                new_sd[f"transformer_blocks.{idx}.attn.norm_k.weight"] = tensor
            elif rest == "txt_attn.qkv.weight":
                q, k, v = tensor.chunk(3, dim=0)
                new_sd[f"transformer_blocks.{idx}.attn.add_q_proj.weight"] = q
                new_sd[f"transformer_blocks.{idx}.attn.add_k_proj.weight"] = k
                new_sd[f"transformer_blocks.{idx}.attn.add_v_proj.weight"] = v
            elif rest == "txt_attn.proj.weight":
                new_sd[f"transformer_blocks.{idx}.attn.to_add_out.weight"] = tensor
            elif rest == "txt_attn.norm.query_norm.scale":
                new_sd[f"transformer_blocks.{idx}.attn.norm_added_q.weight"] = tensor
            elif rest == "txt_attn.norm.key_norm.scale":
                new_sd[f"transformer_blocks.{idx}.attn.norm_added_k.weight"] = tensor
            elif rest == "img_mlp.0.weight":
                new_sd[f"transformer_blocks.{idx}.ff.linear_in.weight"] = tensor
            elif rest == "img_mlp.2.weight":
                new_sd[f"transformer_blocks.{idx}.ff.linear_out.weight"] = tensor
            elif rest == "txt_mlp.0.weight":
                new_sd[f"transformer_blocks.{idx}.ff_context.linear_in.weight"] = tensor
            elif rest == "txt_mlp.2.weight":
                new_sd[
                    f"transformer_blocks.{idx}.ff_context.linear_out.weight"
                ] = tensor

        elif key.startswith("single_blocks."):
            parts = key.split(".")
            idx = parts[1]
            rest = ".".join(parts[2:])
            if rest == "linear1.weight":
                new_sd[
                    f"single_transformer_blocks.{idx}.attn.to_qkv_mlp_proj.weight"
                ] = tensor
            elif rest == "linear2.weight":
                new_sd[f"single_transformer_blocks.{idx}.attn.to_out.weight"] = tensor
            elif rest == "norm.query_norm.scale":
                new_sd[f"single_transformer_blocks.{idx}.attn.norm_q.weight"] = tensor
            elif rest == "norm.key_norm.scale":
                new_sd[f"single_transformer_blocks.{idx}.attn.norm_k.weight"] = tensor

        elif key == "img_in.weight":
            new_sd["x_embedder.weight"] = tensor
        elif key == "txt_in.weight":
            new_sd["context_embedder.weight"] = tensor
        elif key == "time_in.in_layer.weight":
            new_sd["time_guidance_embed.timestep_embedder.linear_1.weight"] = tensor
        elif key == "time_in.out_layer.weight":
            new_sd["time_guidance_embed.timestep_embedder.linear_2.weight"] = tensor
        elif key == "double_stream_modulation_img.lin.weight":
            new_sd["double_stream_modulation_img.linear.weight"] = tensor
        elif key == "double_stream_modulation_txt.lin.weight":
            new_sd["double_stream_modulation_txt.linear.weight"] = tensor
        elif key == "single_stream_modulation.lin.weight":
            new_sd["single_stream_modulation.linear.weight"] = tensor
        elif key == "final_layer.adaLN_modulation.1.weight":
            new_sd["norm_out.linear.weight"] = tensor
        elif key == "final_layer.linear.weight":
            new_sd["proj_out.weight"] = tensor

    return new_sd


class ModelVariant(StrEnum):
    """Available Dark Beast LoRA variants."""

    DARK_BEAST = "DarkBeast"


class ModelLoader(ForgeModel):
    """Dark Beast FLUX.2 Klein LoRA model loader."""

    _VARIANTS = {
        ModelVariant.DARK_BEAST: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DARK_BEAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[Flux2KleinPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DARK_BEAST",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the FLUX.2 Klein pipeline with Dark Beast LoRA weights applied.

        Returns:
            The FLUX.2 Klein transformer model with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = Flux2KleinPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        checkpoint_path = hf_hub_download(LORA_REPO, filename=LORA_WEIGHT_NAME)
        raw_sd = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw_sd[key] = f.get_tensor(key).to(dtype)

        converted_sd = _convert_comfyui_to_diffusers(raw_sd)
        self.pipeline.transformer.load_state_dict(converted_sd, strict=False)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the FLUX.2 Klein transformer.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.pipeline.transformer.config

        # Image dimensions
        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        # Prepare latents: VAE compresses by vae_scale_factor, then pack 2x2 patches
        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        # Create latent tensor (B, C, H, W) then pack to (B, H*W, C)
        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )

        # Prepare latent image IDs (B, H*W, 4)
        t = torch.arange(1)
        h = torch.arange(h_packed)
        w = torch.arange(w_packed)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        # Pack latents: (B, C, H, W) -> (B, H*W, C)
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Prompt embeddings: use random tensors matching joint_attention_dim
        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        # Text IDs (B, seq_len, 4)
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(max_sequence_length)
        text_ids = torch.cartesian_prod(t, h, w, l)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
