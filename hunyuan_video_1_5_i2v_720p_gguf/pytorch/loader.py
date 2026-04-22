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

import json
import os
import tempfile
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

# Small spatial/temporal dimensions for compile-only testing.
TRANSFORMER_NUM_FRAMES = 4
TRANSFORMER_HEIGHT = 8
TRANSFORMER_WIDTH = 8
TRANSFORMER_TEXT_SEQ_LEN = 16
TRANSFORMER_TEXT_SEQ_LEN_2 = 16
TRANSFORMER_IMAGE_SEQ_LEN = 16

# Default config for HunyuanVideo15Transformer3DModel (720p I2V variant).
_HV15_TRANSFORMER_CONFIG = {
    "_class_name": "HunyuanVideo15Transformer3DModel",
    "_diffusers_version": "0.38.0.dev0",
    "in_channels": 65,
    "out_channels": 32,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 54,
    "num_refiner_layers": 2,
    "mlp_ratio": 4.0,
    "patch_size": 1,
    "patch_size_t": 1,
    "qk_norm": "rms_norm",
    "text_embed_dim": 3584,
    "text_embed_2_dim": 1472,
    "image_embed_dim": 1152,
    "rope_theta": 256.0,
    "rope_axes_dim": [16, 56, 56],
    "target_size": 640,
    "task_type": "i2v",
    "use_meanflow": False,
}


def _convert_hunyuan_video_15_i2v_transformer_to_diffusers(checkpoint, **kwargs):
    """Convert HunyuanVideo 1.5 I2V GGUF checkpoint keys to diffusers format.

    Maps from the original Tencent model's naming convention to the diffusers
    HunyuanVideo15Transformer3DModel parameter names. Handles QKV splitting
    for combined attention projections and the scale/shift swap for the final
    AdaLayerNorm modulation.
    """
    new_state_dict = {}

    for key in list(checkpoint.keys()):
        value = checkpoint[key]

        # img_in -> x_embedder
        if key.startswith("img_in.proj."):
            new_key = key.replace("img_in.proj.", "x_embedder.proj.")
            new_state_dict[new_key] = value

        # vision_in -> image_embedder (Sequential: norm_in, linear_1, GELU, linear_2, norm_out)
        elif key.startswith("vision_in.proj."):
            suffix = key[len("vision_in.proj.") :]
            mapping = {
                "0.weight": "image_embedder.norm_in.weight",
                "0.bias": "image_embedder.norm_in.bias",
                "1.weight": "image_embedder.linear_1.weight",
                "1.bias": "image_embedder.linear_1.bias",
                "3.weight": "image_embedder.linear_2.weight",
                "3.bias": "image_embedder.linear_2.bias",
                "4.weight": "image_embedder.norm_out.weight",
                "4.bias": "image_embedder.norm_out.bias",
            }
            if suffix in mapping:
                new_state_dict[mapping[suffix]] = value

        # byt5_in -> context_embedder_2
        elif key.startswith("byt5_in."):
            new_key = (
                key.replace("byt5_in.layernorm.", "context_embedder_2.norm.")
                .replace("byt5_in.fc1.", "context_embedder_2.linear_1.")
                .replace("byt5_in.fc2.", "context_embedder_2.linear_2.")
                .replace("byt5_in.fc3.", "context_embedder_2.linear_3.")
            )
            new_state_dict[new_key] = value

        # time_in -> time_embed
        elif key.startswith("time_in.mlp."):
            suffix = key[len("time_in.mlp.") :]
            mapping = {
                "0.weight": "time_embed.timestep_embedder.linear_1.weight",
                "0.bias": "time_embed.timestep_embedder.linear_1.bias",
                "2.weight": "time_embed.timestep_embedder.linear_2.weight",
                "2.bias": "time_embed.timestep_embedder.linear_2.bias",
            }
            if suffix in mapping:
                new_state_dict[mapping[suffix]] = value

        # cond_type_embedding -> cond_type_embed
        elif key == "cond_type_embedding.weight":
            new_state_dict["cond_type_embed.weight"] = value

        # txt_in -> context_embedder
        elif key.startswith("txt_in."):
            new_key = key
            # t_embedder -> time_text_embed.timestep_embedder
            new_key = new_key.replace(
                "txt_in.t_embedder.mlp.0.",
                "context_embedder.time_text_embed.timestep_embedder.linear_1.",
            )
            new_key = new_key.replace(
                "txt_in.t_embedder.mlp.2.",
                "context_embedder.time_text_embed.timestep_embedder.linear_2.",
            )
            # c_embedder -> time_text_embed.text_embedder
            new_key = new_key.replace(
                "txt_in.c_embedder.linear_1.",
                "context_embedder.time_text_embed.text_embedder.linear_1.",
            )
            new_key = new_key.replace(
                "txt_in.c_embedder.linear_2.",
                "context_embedder.time_text_embed.text_embedder.linear_2.",
            )
            # input_embedder -> proj_in
            new_key = new_key.replace(
                "txt_in.input_embedder.", "context_embedder.proj_in."
            )
            # individual_token_refiner -> token_refiner
            new_key = new_key.replace(
                "txt_in.individual_token_refiner.blocks.",
                "context_embedder.token_refiner.refiner_blocks.",
            )
            # adaLN_modulation.1 -> norm_out.linear (no scale/shift swap for HunyuanVideo15AdaNorm)
            new_key = new_key.replace(".adaLN_modulation.1.", ".norm_out.linear.")
            # mlp -> ff
            new_key = new_key.replace(".mlp.fc1.", ".ff.net.0.proj.")
            new_key = new_key.replace(".mlp.fc2.", ".ff.net.2.")
            # self_attn_proj -> attn.to_out.0
            new_key = new_key.replace(".self_attn_proj.", ".attn.to_out.0.")

            if ".self_attn_qkv." in new_key:
                # Split QKV into separate Q, K, V projections
                q, k, v = value.chunk(3, dim=0)
                base = new_key.replace(".self_attn_qkv.", ".")
                param = base.split(".")[-1]  # weight or bias
                prefix = ".".join(base.split(".")[:-1])
                new_state_dict[f"{prefix}.attn.to_q.{param}"] = q
                new_state_dict[f"{prefix}.attn.to_k.{param}"] = k
                new_state_dict[f"{prefix}.attn.to_v.{param}"] = v
            else:
                new_state_dict[new_key] = value

        # double_blocks -> transformer_blocks
        elif key.startswith("double_blocks."):
            parts = key.split(".")
            block_idx = parts[1]
            rest = ".".join(parts[2:])
            prefix = f"transformer_blocks.{block_idx}"

            if rest.startswith("img_mod.linear."):
                param = rest[len("img_mod.linear.") :]
                new_state_dict[f"{prefix}.norm1.linear.{param}"] = value

            elif rest.startswith("txt_mod.linear."):
                param = rest[len("txt_mod.linear.") :]
                new_state_dict[f"{prefix}.norm1_context.linear.{param}"] = value

            elif rest == "img_attn_q_norm.weight":
                new_state_dict[f"{prefix}.attn.norm_q.weight"] = value

            elif rest == "img_attn_k_norm.weight":
                new_state_dict[f"{prefix}.attn.norm_k.weight"] = value

            elif rest == "txt_attn_q_norm.weight":
                new_state_dict[f"{prefix}.attn.norm_added_q.weight"] = value

            elif rest == "txt_attn_k_norm.weight":
                new_state_dict[f"{prefix}.attn.norm_added_k.weight"] = value

            elif rest.startswith("img_attn_qkv."):
                param = rest[len("img_attn_qkv.") :]
                q, k, v = value.chunk(3, dim=0)
                new_state_dict[f"{prefix}.attn.to_q.{param}"] = q
                new_state_dict[f"{prefix}.attn.to_k.{param}"] = k
                new_state_dict[f"{prefix}.attn.to_v.{param}"] = v

            elif rest.startswith("txt_attn_qkv."):
                param = rest[len("txt_attn_qkv.") :]
                q, k, v = value.chunk(3, dim=0)
                new_state_dict[f"{prefix}.attn.add_q_proj.{param}"] = q
                new_state_dict[f"{prefix}.attn.add_k_proj.{param}"] = k
                new_state_dict[f"{prefix}.attn.add_v_proj.{param}"] = v

            elif rest.startswith("img_attn_proj."):
                param = rest[len("img_attn_proj.") :]
                new_state_dict[f"{prefix}.attn.to_out.0.{param}"] = value

            elif rest.startswith("txt_attn_proj."):
                param = rest[len("txt_attn_proj.") :]
                new_state_dict[f"{prefix}.attn.to_add_out.{param}"] = value

            elif rest.startswith("img_mlp.fc1."):
                param = rest[len("img_mlp.fc1.") :]
                new_state_dict[f"{prefix}.ff.net.0.proj.{param}"] = value

            elif rest.startswith("img_mlp.fc2."):
                param = rest[len("img_mlp.fc2.") :]
                new_state_dict[f"{prefix}.ff.net.2.{param}"] = value

            elif rest.startswith("txt_mlp.fc1."):
                param = rest[len("txt_mlp.fc1.") :]
                new_state_dict[f"{prefix}.ff_context.net.0.proj.{param}"] = value

            elif rest.startswith("txt_mlp.fc2."):
                param = rest[len("txt_mlp.fc2.") :]
                new_state_dict[f"{prefix}.ff_context.net.2.{param}"] = value

        # final_layer -> norm_out / proj_out
        elif key.startswith("final_layer.adaLN_modulation.1."):
            # Swap shift/scale ordering: original [shift, scale] -> diffusers [scale, shift]
            param = key[len("final_layer.adaLN_modulation.1.") :]
            shift, scale = value.chunk(2, dim=0)
            new_state_dict[f"norm_out.linear.{param}"] = torch.cat(
                [scale, shift], dim=0
            )

        elif key.startswith("final_layer.linear."):
            param = key[len("final_layer.linear.") :]
            new_state_dict[f"proj_out.{param}"] = value

        else:
            new_state_dict[key] = value

    return new_state_dict


def _register_hv15_single_file_support():
    """Monkey-patch SINGLE_FILE_LOADABLE_CLASSES to support HunyuanVideo15Transformer3DModel."""
    from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES

    if "HunyuanVideo15Transformer3DModel" not in SINGLE_FILE_LOADABLE_CLASSES:
        SINGLE_FILE_LOADABLE_CLASSES["HunyuanVideo15Transformer3DModel"] = {
            "checkpoint_mapping_fn": _convert_hunyuan_video_15_i2v_transformer_to_diffusers,
            "default_subfolder": "transformer",
        }


def _make_hv15_config_dir():
    """Create a temporary directory with HunyuanVideo15 transformer config.json."""
    tmp_dir = tempfile.mkdtemp(prefix="hv15_config_")
    transformer_dir = os.path.join(tmp_dir, "transformer")
    os.makedirs(transformer_dir)
    config_path = os.path.join(transformer_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(_HV15_TRANSFORMER_CONFIG, f)
    return tmp_dir


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

        # Register HunyuanVideo15Transformer3DModel in diffusers' single-file loader
        # since it is not yet supported natively as of diffusers 0.38.0.dev0.
        _register_hv15_single_file_support()

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        config_dir = _make_hv15_config_dir()
        try:
            self._transformer = HunyuanVideo15Transformer3DModel.from_single_file(
                f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
                config=config_dir,
                subfolder="transformer",
            )
        finally:
            import shutil

            shutil.rmtree(config_dir, ignore_errors=True)

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
