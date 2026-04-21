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
GGUF_BASE_URL = f"https://huggingface.co/{GGUF_REPO}/blob/main"

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


def _convert_hunyuan_video_1_5_gguf_to_diffusers(checkpoint, **kwargs):
    """Convert HunyuanVideo 1.5 GGUF checkpoint keys to diffusers format.

    Maps original architecture key names to diffusers HunyuanVideo15Transformer3DModel
    parameter names and splits fused QKV projections into separate Q/K/V tensors.
    """
    new_checkpoint = {}

    for key in list(checkpoint.keys()):
        value = checkpoint[key]

        if key.startswith("img_in."):
            new_checkpoint[key.replace("img_in.", "x_embedder.")] = value

        elif key.startswith("byt5_in."):
            new_key = (
                key.replace("byt5_in.layernorm.", "context_embedder_2.norm.")
                .replace("byt5_in.fc1.", "context_embedder_2.linear_1.")
                .replace("byt5_in.fc2.", "context_embedder_2.linear_2.")
                .replace("byt5_in.fc3.", "context_embedder_2.linear_3.")
            )
            new_checkpoint[new_key] = value

        elif key == "cond_type_embedding.weight":
            new_checkpoint["cond_type_embed.weight"] = value

        elif key.startswith("time_in."):
            new_key = key.replace(
                "time_in.mlp.0.", "time_embed.timestep_embedder.linear_1."
            ).replace("time_in.mlp.2.", "time_embed.timestep_embedder.linear_2.")
            new_checkpoint[new_key] = value

        elif key.startswith("vision_in."):
            # vision_in uses a Sequential: norm_in(0), linear_1(1), GELU, linear_2(3), norm_out(4)
            new_key = (
                key.replace("vision_in.proj.0.", "image_embedder.norm_in.")
                .replace("vision_in.proj.1.", "image_embedder.linear_1.")
                .replace("vision_in.proj.3.", "image_embedder.linear_2.")
                .replace("vision_in.proj.4.", "image_embedder.norm_out.")
            )
            new_checkpoint[new_key] = value

        elif key.startswith("txt_in."):
            new_key = (
                key.replace(
                    "txt_in.t_embedder.mlp.0.",
                    "context_embedder.time_text_embed.timestep_embedder.linear_1.",
                )
                .replace(
                    "txt_in.t_embedder.mlp.2.",
                    "context_embedder.time_text_embed.timestep_embedder.linear_2.",
                )
                .replace(
                    "txt_in.c_embedder.linear_1.",
                    "context_embedder.time_text_embed.text_embedder.linear_1.",
                )
                .replace(
                    "txt_in.c_embedder.linear_2.",
                    "context_embedder.time_text_embed.text_embedder.linear_2.",
                )
                .replace("txt_in.input_embedder.", "context_embedder.proj_in.")
                .replace(
                    "txt_in.individual_token_refiner.blocks.",
                    "context_embedder.token_refiner.refiner_blocks.",
                )
                .replace(".adaLN_modulation.1.", ".norm_out.linear.")
                .replace(".mlp.fc1.", ".ff.net.0.proj.")
                .replace(".mlp.fc2.", ".ff.net.2.")
                .replace(".self_attn_proj.", ".attn.to_out.0.")
            )
            if ".self_attn_qkv." in new_key:
                to_q, to_k, to_v = value.chunk(3, dim=0)
                new_checkpoint[new_key.replace(".self_attn_qkv.", ".attn.to_q.")] = to_q
                new_checkpoint[new_key.replace(".self_attn_qkv.", ".attn.to_k.")] = to_k
                new_checkpoint[new_key.replace(".self_attn_qkv.", ".attn.to_v.")] = to_v
            else:
                new_checkpoint[new_key] = value

        elif key.startswith("double_blocks."):
            new_key = key.replace("double_blocks.", "transformer_blocks.")
            if ".img_attn_qkv." in new_key:
                to_q, to_k, to_v = value.chunk(3, dim=0)
                new_checkpoint[new_key.replace(".img_attn_qkv.", ".attn.to_q.")] = to_q
                new_checkpoint[new_key.replace(".img_attn_qkv.", ".attn.to_k.")] = to_k
                new_checkpoint[new_key.replace(".img_attn_qkv.", ".attn.to_v.")] = to_v
            elif ".txt_attn_qkv." in new_key:
                to_q, to_k, to_v = value.chunk(3, dim=0)
                new_checkpoint[
                    new_key.replace(".txt_attn_qkv.", ".attn.add_q_proj.")
                ] = to_q
                new_checkpoint[
                    new_key.replace(".txt_attn_qkv.", ".attn.add_k_proj.")
                ] = to_k
                new_checkpoint[
                    new_key.replace(".txt_attn_qkv.", ".attn.add_v_proj.")
                ] = to_v
            else:
                new_key = (
                    new_key.replace(".img_attn_proj.", ".attn.to_out.0.")
                    .replace(".img_attn_q_norm.", ".attn.norm_q.")
                    .replace(".img_attn_k_norm.", ".attn.norm_k.")
                    .replace(".img_mlp.fc1.", ".ff.net.0.proj.")
                    .replace(".img_mlp.fc2.", ".ff.net.2.")
                    .replace(".img_mod.linear.", ".norm1.linear.")
                    .replace(".txt_attn_proj.", ".attn.to_add_out.")
                    .replace(".txt_attn_q_norm.", ".attn.norm_added_q.")
                    .replace(".txt_attn_k_norm.", ".attn.norm_added_k.")
                    .replace(".txt_mlp.fc1.", ".ff_context.net.0.proj.")
                    .replace(".txt_mlp.fc2.", ".ff_context.net.2.")
                    .replace(".txt_mod.linear.", ".norm1_context.linear.")
                )
                new_checkpoint[new_key] = value

        elif key.startswith("final_layer."):
            new_key = key.replace(
                "final_layer.adaLN_modulation.1.", "norm_out.linear."
            ).replace("final_layer.linear.", "proj_out.")
            new_checkpoint[new_key] = value

        else:
            new_checkpoint[key] = value

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

        from diffusers import (
            GGUFQuantizationConfig,
            HunyuanVideo15Transformer3DModel,
        )
        from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES

        # HunyuanVideo15Transformer3DModel is not in SINGLE_FILE_LOADABLE_CLASSES in
        # diffusers 0.37.x; register it with the key-conversion function required to
        # map original HunyuanVideo 1.5 GGUF architecture names to diffusers names.
        if "HunyuanVideo15Transformer3DModel" not in SINGLE_FILE_LOADABLE_CLASSES:
            SINGLE_FILE_LOADABLE_CLASSES["HunyuanVideo15Transformer3DModel"] = {
                "checkpoint_mapping_fn": _convert_hunyuan_video_1_5_gguf_to_diffusers,
                "default_subfolder": "transformer",
            }

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = HunyuanVideo15Transformer3DModel.from_single_file(
            f"{GGUF_BASE_URL}/{gguf_file}",
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
