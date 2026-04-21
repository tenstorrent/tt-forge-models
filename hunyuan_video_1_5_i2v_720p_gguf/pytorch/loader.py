#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanVideo 1.5 I2V 720p GGUF model loader implementation.

Loads GGUF-quantized HunyuanVideo 1.5 image-to-video transformers from
jayn7/HunyuanVideo-1.5_I2V_720p-GGUF. The base model is Tencent's
HunyuanVideo 1.5 (~8B parameter DiT) repackaged in multiple quantization
levels (Q4_K_S/M, Q5_K_S/M, Q6_K, Q8_0) for both the standard 720p
checkpoint and the cfg-distilled 720p variant.

Repository:
- https://huggingface.co/jayn7/HunyuanVideo-1.5_I2V_720p-GGUF
"""

import os
from typing import Any, Optional

import torch

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

GGUF_REPO = "jayn7/HunyuanVideo-1.5_I2V_720p-GGUF"
_CONFIG_DIR = os.path.dirname(__file__)

# Small spatial/temporal dimensions for compile-only testing.
TRANSFORMER_NUM_FRAMES = 4
TRANSFORMER_HEIGHT = 8
TRANSFORMER_WIDTH = 8
TRANSFORMER_TEXT_SEQ_LEN = 16
TRANSFORMER_TEXT_SEQ_LEN_2 = 16
TRANSFORMER_IMAGE_SEQ_LEN = 16


class ModelVariant(StrEnum):
    """Available HunyuanVideo 1.5 I2V 720p GGUF variants."""

    I2V_720P_Q4_K_S = "720p_i2v_Q4_K_S"
    I2V_720P_Q4_K_M = "720p_i2v_Q4_K_M"
    I2V_720P_Q5_K_S = "720p_i2v_Q5_K_S"
    I2V_720P_Q5_K_M = "720p_i2v_Q5_K_M"
    I2V_720P_Q6_K = "720p_i2v_Q6_K"
    I2V_720P_Q8_0 = "720p_i2v_Q8_0"
    I2V_720P_CFG_DISTILLED_Q4_K_S = "720p_i2v_cfg_distilled_Q4_K_S"
    I2V_720P_CFG_DISTILLED_Q4_K_M = "720p_i2v_cfg_distilled_Q4_K_M"
    I2V_720P_CFG_DISTILLED_Q5_K_S = "720p_i2v_cfg_distilled_Q5_K_S"
    I2V_720P_CFG_DISTILLED_Q5_K_M = "720p_i2v_cfg_distilled_Q5_K_M"
    I2V_720P_CFG_DISTILLED_Q6_K = "720p_i2v_cfg_distilled_Q6_K"
    I2V_720P_CFG_DISTILLED_Q8_0 = "720p_i2v_cfg_distilled_Q8_0"


_GGUF_FILES = {
    ModelVariant.I2V_720P_Q4_K_S: "720p/hunyuanvideo1.5_720p_i2v-Q4_K_S.gguf",
    ModelVariant.I2V_720P_Q4_K_M: "720p/hunyuanvideo1.5_720p_i2v-Q4_K_M.gguf",
    ModelVariant.I2V_720P_Q5_K_S: "720p/hunyuanvideo1.5_720p_i2v-Q5_K_S.gguf",
    ModelVariant.I2V_720P_Q5_K_M: "720p/hunyuanvideo1.5_720p_i2v-Q5_K_M.gguf",
    ModelVariant.I2V_720P_Q6_K: "720p/hunyuanvideo1.5_720p_i2v-Q6_K.gguf",
    ModelVariant.I2V_720P_Q8_0: "720p/hunyuanvideo1.5_720p_i2v-Q8_0.gguf",
    ModelVariant.I2V_720P_CFG_DISTILLED_Q4_K_S: "720p_distilled/hunyuanvideo1.5_720p_i2v_cfg_distilled-Q4_K_S.gguf",
    ModelVariant.I2V_720P_CFG_DISTILLED_Q4_K_M: "720p_distilled/hunyuanvideo1.5_720p_i2v_cfg_distilled-Q4_K_M.gguf",
    ModelVariant.I2V_720P_CFG_DISTILLED_Q5_K_S: "720p_distilled/hunyuanvideo1.5_720p_i2v_cfg_distilled-Q5_K_S.gguf",
    ModelVariant.I2V_720P_CFG_DISTILLED_Q5_K_M: "720p_distilled/hunyuanvideo1.5_720p_i2v_cfg_distilled-Q5_K_M.gguf",
    ModelVariant.I2V_720P_CFG_DISTILLED_Q6_K: "720p_distilled/hunyuanvideo1.5_720p_i2v_cfg_distilled-Q6_K.gguf",
    ModelVariant.I2V_720P_CFG_DISTILLED_Q8_0: "720p_distilled/hunyuanvideo1.5_720p_i2v_cfg_distilled-Q8_0.gguf",
}


def _convert_hunyuan_video15_transformer_to_diffusers(checkpoint, **kwargs):
    """Convert HunyuanVideo 1.5 GGUF checkpoint from original to diffusers format.

    Maps the original HunyuanVideo 1.5 I2V key naming convention used in GGUF
    files to the diffusers HunyuanVideo15Transformer3DModel key naming convention.
    """
    new_ckpt = {}

    def _rename(old_key, new_key):
        if old_key in checkpoint:
            new_ckpt[new_key] = checkpoint.pop(old_key)

    def _split_qkv(old_key, q_key, k_key, v_key):
        if old_key not in checkpoint:
            return
        weight = checkpoint.pop(old_key)
        q, k, v = weight.chunk(3, dim=0)
        new_ckpt[q_key] = q
        new_ckpt[k_key] = k
        new_ckpt[v_key] = v

    # x_embedder
    _rename("img_in.proj.weight", "x_embedder.proj.weight")
    _rename("img_in.proj.bias", "x_embedder.proj.bias")

    # cond_type_embed
    _rename("cond_type_embedding.weight", "cond_type_embed.weight")

    # time_embed
    _rename("time_in.mlp.0.weight", "time_embed.timestep_embedder.linear_1.weight")
    _rename("time_in.mlp.0.bias", "time_embed.timestep_embedder.linear_1.bias")
    _rename("time_in.mlp.2.weight", "time_embed.timestep_embedder.linear_2.weight")
    _rename("time_in.mlp.2.bias", "time_embed.timestep_embedder.linear_2.bias")

    # image_embedder (vision_in)
    _rename("vision_in.proj.0.weight", "image_embedder.norm_in.weight")
    _rename("vision_in.proj.0.bias", "image_embedder.norm_in.bias")
    _rename("vision_in.proj.1.weight", "image_embedder.linear_1.weight")
    _rename("vision_in.proj.1.bias", "image_embedder.linear_1.bias")
    _rename("vision_in.proj.3.weight", "image_embedder.linear_2.weight")
    _rename("vision_in.proj.3.bias", "image_embedder.linear_2.bias")
    _rename("vision_in.proj.4.weight", "image_embedder.norm_out.weight")
    _rename("vision_in.proj.4.bias", "image_embedder.norm_out.bias")

    # context_embedder (txt_in)
    _rename(
        "txt_in.t_embedder.mlp.0.weight",
        "context_embedder.time_text_embed.timestep_embedder.linear_1.weight",
    )
    _rename(
        "txt_in.t_embedder.mlp.0.bias",
        "context_embedder.time_text_embed.timestep_embedder.linear_1.bias",
    )
    _rename(
        "txt_in.t_embedder.mlp.2.weight",
        "context_embedder.time_text_embed.timestep_embedder.linear_2.weight",
    )
    _rename(
        "txt_in.t_embedder.mlp.2.bias",
        "context_embedder.time_text_embed.timestep_embedder.linear_2.bias",
    )
    _rename(
        "txt_in.c_embedder.linear_1.weight",
        "context_embedder.time_text_embed.text_embedder.linear_1.weight",
    )
    _rename(
        "txt_in.c_embedder.linear_1.bias",
        "context_embedder.time_text_embed.text_embedder.linear_1.bias",
    )
    _rename(
        "txt_in.c_embedder.linear_2.weight",
        "context_embedder.time_text_embed.text_embedder.linear_2.weight",
    )
    _rename(
        "txt_in.c_embedder.linear_2.bias",
        "context_embedder.time_text_embed.text_embedder.linear_2.bias",
    )
    _rename("txt_in.input_embedder.weight", "context_embedder.proj_in.weight")
    _rename("txt_in.input_embedder.bias", "context_embedder.proj_in.bias")

    # token_refiner blocks
    refiner_prefix = "txt_in.individual_token_refiner.blocks"
    diffusers_refiner_prefix = "context_embedder.token_refiner.refiner_blocks"
    block_idx = 0
    while f"{refiner_prefix}.{block_idx}.norm1.weight" in checkpoint:
        n = block_idx
        src = f"{refiner_prefix}.{n}"
        dst = f"{diffusers_refiner_prefix}.{n}"
        _rename(f"{src}.norm1.weight", f"{dst}.norm1.weight")
        _rename(f"{src}.norm1.bias", f"{dst}.norm1.bias")
        _rename(f"{src}.norm2.weight", f"{dst}.norm2.weight")
        _rename(f"{src}.norm2.bias", f"{dst}.norm2.bias")
        _split_qkv(
            f"{src}.self_attn_qkv.weight",
            f"{dst}.attn.to_q.weight",
            f"{dst}.attn.to_k.weight",
            f"{dst}.attn.to_v.weight",
        )
        _split_qkv(
            f"{src}.self_attn_qkv.bias",
            f"{dst}.attn.to_q.bias",
            f"{dst}.attn.to_k.bias",
            f"{dst}.attn.to_v.bias",
        )
        _rename(f"{src}.self_attn_proj.weight", f"{dst}.attn.to_out.0.weight")
        _rename(f"{src}.self_attn_proj.bias", f"{dst}.attn.to_out.0.bias")
        _rename(f"{src}.mlp.fc1.weight", f"{dst}.ff.net.0.proj.weight")
        _rename(f"{src}.mlp.fc1.bias", f"{dst}.ff.net.0.proj.bias")
        _rename(f"{src}.mlp.fc2.weight", f"{dst}.ff.net.2.weight")
        _rename(f"{src}.mlp.fc2.bias", f"{dst}.ff.net.2.bias")
        _rename(f"{src}.adaLN_modulation.1.weight", f"{dst}.norm_out.linear.weight")
        _rename(f"{src}.adaLN_modulation.1.bias", f"{dst}.norm_out.linear.bias")
        block_idx += 1

    # context_embedder_2 (byt5_in)
    _rename("byt5_in.layernorm.weight", "context_embedder_2.norm.weight")
    _rename("byt5_in.layernorm.bias", "context_embedder_2.norm.bias")
    _rename("byt5_in.fc1.weight", "context_embedder_2.linear_1.weight")
    _rename("byt5_in.fc1.bias", "context_embedder_2.linear_1.bias")
    _rename("byt5_in.fc2.weight", "context_embedder_2.linear_2.weight")
    _rename("byt5_in.fc2.bias", "context_embedder_2.linear_2.bias")
    _rename("byt5_in.fc3.weight", "context_embedder_2.linear_3.weight")
    _rename("byt5_in.fc3.bias", "context_embedder_2.linear_3.bias")

    # transformer_blocks (double_blocks)
    block_idx = 0
    while f"double_blocks.{block_idx}.img_attn_proj.weight" in checkpoint:
        n = block_idx
        src = f"double_blocks.{n}"
        dst = f"transformer_blocks.{n}"
        _rename(f"{src}.img_mod.linear.weight", f"{dst}.norm1.linear.weight")
        _rename(f"{src}.img_mod.linear.bias", f"{dst}.norm1.linear.bias")
        _rename(f"{src}.txt_mod.linear.weight", f"{dst}.norm1_context.linear.weight")
        _rename(f"{src}.txt_mod.linear.bias", f"{dst}.norm1_context.linear.bias")
        _split_qkv(
            f"{src}.img_attn_qkv.weight",
            f"{dst}.attn.to_q.weight",
            f"{dst}.attn.to_k.weight",
            f"{dst}.attn.to_v.weight",
        )
        _split_qkv(
            f"{src}.img_attn_qkv.bias",
            f"{dst}.attn.to_q.bias",
            f"{dst}.attn.to_k.bias",
            f"{dst}.attn.to_v.bias",
        )
        _split_qkv(
            f"{src}.txt_attn_qkv.weight",
            f"{dst}.attn.add_q_proj.weight",
            f"{dst}.attn.add_k_proj.weight",
            f"{dst}.attn.add_v_proj.weight",
        )
        _split_qkv(
            f"{src}.txt_attn_qkv.bias",
            f"{dst}.attn.add_q_proj.bias",
            f"{dst}.attn.add_k_proj.bias",
            f"{dst}.attn.add_v_proj.bias",
        )
        _rename(f"{src}.img_attn_proj.weight", f"{dst}.attn.to_out.0.weight")
        _rename(f"{src}.img_attn_proj.bias", f"{dst}.attn.to_out.0.bias")
        _rename(f"{src}.txt_attn_proj.weight", f"{dst}.attn.to_add_out.weight")
        _rename(f"{src}.txt_attn_proj.bias", f"{dst}.attn.to_add_out.bias")
        _rename(f"{src}.img_attn_q_norm.weight", f"{dst}.attn.norm_q.weight")
        _rename(f"{src}.img_attn_k_norm.weight", f"{dst}.attn.norm_k.weight")
        _rename(f"{src}.txt_attn_q_norm.weight", f"{dst}.attn.norm_added_q.weight")
        _rename(f"{src}.txt_attn_k_norm.weight", f"{dst}.attn.norm_added_k.weight")
        _rename(f"{src}.img_mlp.fc1.weight", f"{dst}.ff.net.0.proj.weight")
        _rename(f"{src}.img_mlp.fc1.bias", f"{dst}.ff.net.0.proj.bias")
        _rename(f"{src}.img_mlp.fc2.weight", f"{dst}.ff.net.2.weight")
        _rename(f"{src}.img_mlp.fc2.bias", f"{dst}.ff.net.2.bias")
        _rename(f"{src}.txt_mlp.fc1.weight", f"{dst}.ff_context.net.0.proj.weight")
        _rename(f"{src}.txt_mlp.fc1.bias", f"{dst}.ff_context.net.0.proj.bias")
        _rename(f"{src}.txt_mlp.fc2.weight", f"{dst}.ff_context.net.2.weight")
        _rename(f"{src}.txt_mlp.fc2.bias", f"{dst}.ff_context.net.2.bias")
        block_idx += 1

    # norm_out (adaLN final layer: original stores [shift, scale], diffusers wants [scale, shift])
    for suffix in ("weight", "bias"):
        old_key = f"final_layer.adaLN_modulation.1.{suffix}"
        if old_key in checkpoint:
            tensor = checkpoint.pop(old_key)
            shift, scale = tensor.chunk(2, dim=0)
            new_ckpt[f"norm_out.linear.{suffix}"] = torch.cat([scale, shift], dim=0)

    # proj_out
    _rename("final_layer.linear.weight", "proj_out.weight")
    _rename("final_layer.linear.bias", "proj_out.bias")

    return new_ckpt


class ModelLoader(ForgeModel):
    """HunyuanVideo 1.5 I2V 720p GGUF model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }
    DEFAULT_VARIANT = ModelVariant.I2V_720P_Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HunyuanVideo-1.5_I2V_720p_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized HunyuanVideo 1.5 I2V transformer."""
        from diffusers import (
            GGUFQuantizationConfig,
            HunyuanVideo15Transformer3DModel,
        )
        from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES

        # diffusers 0.37.1 is missing HunyuanVideo15Transformer3DModel in
        # SINGLE_FILE_LOADABLE_CLASSES; register it with the v1.5-specific converter.
        if "HunyuanVideo15Transformer3DModel" not in SINGLE_FILE_LOADABLE_CLASSES:
            SINGLE_FILE_LOADABLE_CLASSES["HunyuanVideo15Transformer3DModel"] = {
                "checkpoint_mapping_fn": _convert_hunyuan_video15_transformer_to_diffusers,
            }

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = HunyuanVideo15Transformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            config=_CONFIG_DIR,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare tensor inputs for the HunyuanVideo15Transformer3DModel forward pass."""
        if self._transformer is None:
            self.load_model()

        dtype = torch.bfloat16
        config = self._transformer.config
        batch_size = 1

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            TRANSFORMER_NUM_FRAMES,
            TRANSFORMER_HEIGHT,
            TRANSFORMER_WIDTH,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            TRANSFORMER_TEXT_SEQ_LEN,
            config.text_embed_dim,
            dtype=dtype,
        )
        encoder_attention_mask = torch.ones(
            batch_size, TRANSFORMER_TEXT_SEQ_LEN, dtype=torch.bool
        )
        encoder_hidden_states_2 = torch.randn(
            batch_size,
            TRANSFORMER_TEXT_SEQ_LEN_2,
            config.text_embed_2_dim,
            dtype=dtype,
        )
        encoder_attention_mask_2 = torch.ones(
            batch_size, TRANSFORMER_TEXT_SEQ_LEN_2, dtype=torch.bool
        )
        image_embeds = torch.randn(
            batch_size,
            TRANSFORMER_IMAGE_SEQ_LEN,
            config.image_embed_dim,
            dtype=dtype,
        )
        timestep = torch.tensor([500], dtype=torch.long).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_hidden_states_2": encoder_hidden_states_2,
            "encoder_attention_mask_2": encoder_attention_mask_2,
            "image_embeds": image_embeds,
            "return_dict": False,
        }
