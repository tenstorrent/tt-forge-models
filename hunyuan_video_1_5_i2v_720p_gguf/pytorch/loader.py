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

# Diffusers-format config repo for HunyuanVideo 1.5 I2V 720p
DIFFUSERS_CONFIG_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v"

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
    """Convert HunyuanVideo 1.5 original Tencent format to diffusers format.

    The GGUF files from jayn7/HunyuanVideo-1.5_I2V_720p-GGUF use the original
    Tencent key naming convention. This function remaps them to the naming used
    by diffusers' HunyuanVideo15Transformer3DModel.
    """
    new_checkpoint = {}

    for key, value in checkpoint.items():
        # img_in -> x_embedder
        if key.startswith("img_in."):
            new_checkpoint[key.replace("img_in.", "x_embedder.", 1)] = value

        # txt_in input embedder -> context_embedder.proj_in
        elif key.startswith("txt_in.input_embedder."):
            new_checkpoint[
                key.replace("txt_in.input_embedder.", "context_embedder.proj_in.", 1)
            ] = value

        # txt_in time embedder -> context_embedder.time_text_embed.timestep_embedder
        elif key.startswith("txt_in.t_embedder.mlp.0."):
            new_checkpoint[
                key.replace(
                    "txt_in.t_embedder.mlp.0.",
                    "context_embedder.time_text_embed.timestep_embedder.linear_1.",
                    1,
                )
            ] = value
        elif key.startswith("txt_in.t_embedder.mlp.2."):
            new_checkpoint[
                key.replace(
                    "txt_in.t_embedder.mlp.2.",
                    "context_embedder.time_text_embed.timestep_embedder.linear_2.",
                    1,
                )
            ] = value

        # txt_in text embedder -> context_embedder.time_text_embed.text_embedder
        elif key.startswith("txt_in.c_embedder.linear_1."):
            new_checkpoint[
                key.replace(
                    "txt_in.c_embedder.linear_1.",
                    "context_embedder.time_text_embed.text_embedder.linear_1.",
                    1,
                )
            ] = value
        elif key.startswith("txt_in.c_embedder.linear_2."):
            new_checkpoint[
                key.replace(
                    "txt_in.c_embedder.linear_2.",
                    "context_embedder.time_text_embed.text_embedder.linear_2.",
                    1,
                )
            ] = value

        # txt_in individual token refiner -> context_embedder.token_refiner.refiner_blocks
        elif key.startswith("txt_in.individual_token_refiner.blocks."):
            new_key = key.replace(
                "txt_in.individual_token_refiner.blocks.",
                "context_embedder.token_refiner.refiner_blocks.",
                1,
            )
            new_key = new_key.replace(".adaLN_modulation.1.", ".norm_out.linear.")
            new_key = new_key.replace(".mlp.fc1.", ".ff.net.0.proj.")
            new_key = new_key.replace(".mlp.fc2.", ".ff.net.2.")
            new_key = new_key.replace(".self_attn_proj.", ".attn.to_out.0.")

            if ".self_attn_qkv." in new_key:
                # Split combined qkv into separate q, k, v tensors
                suffix = new_key.split(".self_attn_qkv.")[-1]  # "weight" or "bias"
                base = new_key.split(".self_attn_qkv.")[0]
                q, k, v = value.chunk(3, dim=0)
                new_checkpoint[f"{base}.attn.to_q.{suffix}"] = q
                new_checkpoint[f"{base}.attn.to_k.{suffix}"] = k
                new_checkpoint[f"{base}.attn.to_v.{suffix}"] = v
            else:
                new_checkpoint[new_key] = value

        # double_blocks -> transformer_blocks
        elif key.startswith("double_blocks."):
            new_key = key.replace("double_blocks.", "transformer_blocks.", 1)
            new_key = new_key.replace(".img_attn_q.", ".attn.to_q.")
            new_key = new_key.replace(".img_attn_k.", ".attn.to_k.")
            new_key = new_key.replace(".img_attn_v.", ".attn.to_v.")
            new_key = new_key.replace(".img_attn_proj.", ".attn.to_out.0.")
            new_key = new_key.replace(".img_attn_q_norm.", ".attn.norm_q.")
            new_key = new_key.replace(".img_attn_k_norm.", ".attn.norm_k.")
            new_key = new_key.replace(".img_mlp.fc1.", ".ff.net.0.proj.")
            new_key = new_key.replace(".img_mlp.fc2.", ".ff.net.2.")
            new_key = new_key.replace(".img_mod.linear.", ".norm1.linear.")
            new_key = new_key.replace(".txt_attn_q.", ".attn.add_q_proj.")
            new_key = new_key.replace(".txt_attn_k.", ".attn.add_k_proj.")
            new_key = new_key.replace(".txt_attn_v.", ".attn.add_v_proj.")
            new_key = new_key.replace(".txt_attn_proj.", ".attn.to_add_out.")
            new_key = new_key.replace(".txt_attn_q_norm.", ".attn.norm_added_q.")
            new_key = new_key.replace(".txt_attn_k_norm.", ".attn.norm_added_k.")
            new_key = new_key.replace(".txt_mlp.fc1.", ".ff_context.net.0.proj.")
            new_key = new_key.replace(".txt_mlp.fc2.", ".ff_context.net.2.")
            new_key = new_key.replace(".txt_mod.linear.", ".norm1_context.linear.")
            new_checkpoint[new_key] = value

        # time_in -> time_embed.timestep_embedder
        elif key.startswith("time_in.mlp.0."):
            new_checkpoint[
                key.replace(
                    "time_in.mlp.0.",
                    "time_embed.timestep_embedder.linear_1.",
                    1,
                )
            ] = value
        elif key.startswith("time_in.mlp.2."):
            new_checkpoint[
                key.replace(
                    "time_in.mlp.2.",
                    "time_embed.timestep_embedder.linear_2.",
                    1,
                )
            ] = value

        # final_layer -> norm_out + proj_out
        elif key.startswith("final_layer.adaLN_modulation.1."):
            new_checkpoint[
                key.replace("final_layer.adaLN_modulation.1.", "norm_out.linear.", 1)
            ] = value
        elif key.startswith("final_layer.linear."):
            new_checkpoint[key.replace("final_layer.linear.", "proj_out.", 1)] = value

        # cond_type_embedding -> cond_type_embed
        elif key == "cond_type_embedding.weight":
            new_checkpoint["cond_type_embed.weight"] = value

        # byt5_in -> context_embedder_2 (glyph ByteT5 encoder)
        elif key.startswith("byt5_in."):
            new_key = key
            new_key = new_key.replace("byt5_in.fc1.", "context_embedder_2.linear_1.")
            new_key = new_key.replace("byt5_in.fc2.", "context_embedder_2.linear_2.")
            new_key = new_key.replace("byt5_in.fc3.", "context_embedder_2.linear_3.")
            new_key = new_key.replace("byt5_in.layernorm.", "context_embedder_2.norm.")
            new_checkpoint[new_key] = value

        # vision_in -> image_embedder (image conditioning for I2V)
        # Sequential: [LayerNorm(0), Linear(1), GELU(2), Linear(3), LayerNorm(4)]
        elif key.startswith("vision_in."):
            new_key = key
            new_key = new_key.replace("vision_in.proj.0.", "image_embedder.norm_in.")
            new_key = new_key.replace("vision_in.proj.1.", "image_embedder.linear_1.")
            new_key = new_key.replace("vision_in.proj.3.", "image_embedder.linear_2.")
            new_key = new_key.replace("vision_in.proj.4.", "image_embedder.norm_out.")
            new_checkpoint[new_key] = value

    return new_checkpoint


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
        """Load the GGUF-quantized HunyuanVideo 1.5 I2V transformer.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer.
        Returns the transformer nn.Module directly for compilation testing.
        """
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        import diffusers.loaders.single_file_model as _sfm
        from diffusers import (
            GGUFQuantizationConfig,
            HunyuanVideo15Transformer3DModel,
        )
        from huggingface_hub import hf_hub_download

        # HunyuanVideo15Transformer3DModel is not in SINGLE_FILE_LOADABLE_CLASSES in
        # diffusers 0.37.1. Register it with a conversion function that maps the
        # original Tencent key format to the diffusers naming convention.
        if "HunyuanVideo15Transformer3DModel" not in _sfm.SINGLE_FILE_LOADABLE_CLASSES:
            _sfm.SINGLE_FILE_LOADABLE_CLASSES["HunyuanVideo15Transformer3DModel"] = {
                "checkpoint_mapping_fn": _convert_hunyuan_video15_transformer_to_diffusers,
                "default_subfolder": "transformer",
            }

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = HunyuanVideo15Transformer3DModel.from_single_file(
            gguf_path,
            config=DIFFUSERS_CONFIG_REPO,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=False,
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
