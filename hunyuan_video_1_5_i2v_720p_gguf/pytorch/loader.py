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


def _convert_hunyuan_video_1_5_i2v_transformer_to_diffusers(checkpoint, **kwargs):
    """Convert HunyuanVideo 1.5 I2V GGUF checkpoint keys to diffusers naming.

    The GGUF uses original Tencent naming conventions; this function maps them
    to the diffusers HunyuanVideo15Transformer3DModel naming conventions.
    """

    def remap_norm_scale_shift_(key, state_dict, new_key):
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        state_dict[new_key] = torch.cat([scale, shift], dim=0)

    converted = {}

    for key in list(checkpoint.keys()):
        val = checkpoint[key]

        # Global timestep embedder
        if key == "time_in.mlp.0.weight":
            converted["time_embed.timestep_embedder.linear_1.weight"] = val
        elif key == "time_in.mlp.0.bias":
            converted["time_embed.timestep_embedder.linear_1.bias"] = val
        elif key == "time_in.mlp.2.weight":
            converted["time_embed.timestep_embedder.linear_2.weight"] = val
        elif key == "time_in.mlp.2.bias":
            converted["time_embed.timestep_embedder.linear_2.bias"] = val

        # Image patch embedder
        elif key == "img_in.proj.weight":
            converted["x_embedder.proj.weight"] = val
        elif key == "img_in.proj.bias":
            converted["x_embedder.proj.bias"] = val

        # Conditioning type embedding
        elif key == "cond_type_embedding.weight":
            converted["cond_type_embed.weight"] = val

        # ByT5 text projection (context_embedder_2)
        elif key == "byt5_in.layernorm.weight":
            converted["context_embedder_2.norm.weight"] = val
        elif key == "byt5_in.layernorm.bias":
            converted["context_embedder_2.norm.bias"] = val
        elif key == "byt5_in.fc1.weight":
            converted["context_embedder_2.linear_1.weight"] = val
        elif key == "byt5_in.fc1.bias":
            converted["context_embedder_2.linear_1.bias"] = val
        elif key == "byt5_in.fc2.weight":
            converted["context_embedder_2.linear_2.weight"] = val
        elif key == "byt5_in.fc2.bias":
            converted["context_embedder_2.linear_2.bias"] = val
        elif key == "byt5_in.fc3.weight":
            converted["context_embedder_2.linear_3.weight"] = val
        elif key == "byt5_in.fc3.bias":
            converted["context_embedder_2.linear_3.bias"] = val

        # Image embedder (vision_in → image_embedder)
        elif key == "vision_in.proj.0.weight":
            converted["image_embedder.norm_in.weight"] = val
        elif key == "vision_in.proj.0.bias":
            converted["image_embedder.norm_in.bias"] = val
        elif key == "vision_in.proj.1.weight":
            converted["image_embedder.linear_1.weight"] = val
        elif key == "vision_in.proj.1.bias":
            converted["image_embedder.linear_1.bias"] = val
        elif key == "vision_in.proj.3.weight":
            converted["image_embedder.linear_2.weight"] = val
        elif key == "vision_in.proj.3.bias":
            converted["image_embedder.linear_2.bias"] = val
        elif key == "vision_in.proj.4.weight":
            converted["image_embedder.norm_out.weight"] = val
        elif key == "vision_in.proj.4.bias":
            converted["image_embedder.norm_out.bias"] = val

        # Text context embedder (txt_in → context_embedder)
        elif key == "txt_in.input_embedder.weight":
            converted["context_embedder.proj_in.weight"] = val
        elif key == "txt_in.input_embedder.bias":
            converted["context_embedder.proj_in.bias"] = val
        elif key == "txt_in.c_embedder.linear_1.weight":
            converted[
                "context_embedder.time_text_embed.text_embedder.linear_1.weight"
            ] = val
        elif key == "txt_in.c_embedder.linear_1.bias":
            converted[
                "context_embedder.time_text_embed.text_embedder.linear_1.bias"
            ] = val
        elif key == "txt_in.c_embedder.linear_2.weight":
            converted[
                "context_embedder.time_text_embed.text_embedder.linear_2.weight"
            ] = val
        elif key == "txt_in.c_embedder.linear_2.bias":
            converted[
                "context_embedder.time_text_embed.text_embedder.linear_2.bias"
            ] = val
        elif key == "txt_in.t_embedder.mlp.0.weight":
            converted[
                "context_embedder.time_text_embed.timestep_embedder.linear_1.weight"
            ] = val
        elif key == "txt_in.t_embedder.mlp.0.bias":
            converted[
                "context_embedder.time_text_embed.timestep_embedder.linear_1.bias"
            ] = val
        elif key == "txt_in.t_embedder.mlp.2.weight":
            converted[
                "context_embedder.time_text_embed.timestep_embedder.linear_2.weight"
            ] = val
        elif key == "txt_in.t_embedder.mlp.2.bias":
            converted[
                "context_embedder.time_text_embed.timestep_embedder.linear_2.bias"
            ] = val

        # Output projection
        elif key == "final_layer.linear.weight":
            converted["proj_out.weight"] = val
        elif key == "final_layer.linear.bias":
            converted["proj_out.bias"] = val

        else:
            converted[key] = val

    # Handle final_layer.adaLN_modulation (scale/shift rearrangement)
    for suffix in ("weight", "bias"):
        src = f"final_layer.adaLN_modulation.1.{suffix}"
        if src in converted:
            remap_norm_scale_shift_(src, converted, f"norm_out.linear.{suffix}")

    # Handle txt_in.individual_token_refiner.blocks.N (token refiner)
    for key in list(converted.keys()):
        if "txt_in.individual_token_refiner.blocks." not in key:
            continue
        val = converted.pop(key)
        # txt_in.individual_token_refiner.blocks.N.X → context_embedder.token_refiner.refiner_blocks.N.X
        new_key = key.replace(
            "txt_in.individual_token_refiner.blocks.",
            "context_embedder.token_refiner.refiner_blocks.",
        )
        new_key = new_key.replace("adaLN_modulation.1", "norm_out.linear")
        new_key = new_key.replace("mlp.fc1", "ff.net.0.proj")
        new_key = new_key.replace("mlp.fc2", "ff.net.2")
        new_key = new_key.replace("self_attn_proj", "attn.to_out.0")
        if "self_attn_qkv" in new_key:
            block_prefix = new_key.replace(
                ".self_attn_qkv." + new_key.split(".")[-1], ""
            )
            suffix = new_key.split(".")[-1]
            q, k, v = val.chunk(3, dim=0)
            converted[f"{block_prefix}.attn.to_q.{suffix}"] = q
            converted[f"{block_prefix}.attn.to_k.{suffix}"] = k
            converted[f"{block_prefix}.attn.to_v.{suffix}"] = v
        else:
            converted[new_key] = val

    # Handle double_blocks.N (transformer blocks)
    for key in list(converted.keys()):
        if not key.startswith("double_blocks."):
            continue
        val = converted.pop(key)
        # double_blocks.N.X → transformer_blocks.N.X
        new_key = key.replace("double_blocks.", "transformer_blocks.")
        new_key = new_key.replace("img_attn_q_norm", "attn.norm_q")
        new_key = new_key.replace("img_attn_k_norm", "attn.norm_k")
        new_key = new_key.replace("img_attn_proj", "attn.to_out.0")
        new_key = new_key.replace("txt_attn_q_norm", "attn.norm_added_q")
        new_key = new_key.replace("txt_attn_k_norm", "attn.norm_added_k")
        new_key = new_key.replace("txt_attn_proj", "attn.to_add_out")
        new_key = new_key.replace("img_mod.linear", "norm1.linear")
        new_key = new_key.replace("txt_mod.linear", "norm1_context.linear")
        new_key = new_key.replace("img_mlp.fc1", "ff.net.0.proj")
        new_key = new_key.replace("img_mlp.fc2", "ff.net.2")
        new_key = new_key.replace("txt_mlp.fc1", "ff_context.net.0.proj")
        new_key = new_key.replace("txt_mlp.fc2", "ff_context.net.2")

        if "img_attn_qkv" in new_key:
            block_n = key.split(".")[1]
            suffix = key.split(".")[-1]
            q, k, v = val.chunk(3, dim=0)
            converted[f"transformer_blocks.{block_n}.attn.to_q.{suffix}"] = q
            converted[f"transformer_blocks.{block_n}.attn.to_k.{suffix}"] = k
            converted[f"transformer_blocks.{block_n}.attn.to_v.{suffix}"] = v
        elif "txt_attn_qkv" in new_key:
            block_n = key.split(".")[1]
            suffix = key.split(".")[-1]
            q, k, v = val.chunk(3, dim=0)
            converted[f"transformer_blocks.{block_n}.attn.add_q_proj.{suffix}"] = q
            converted[f"transformer_blocks.{block_n}.attn.add_k_proj.{suffix}"] = k
            converted[f"transformer_blocks.{block_n}.attn.add_v_proj.{suffix}"] = v
        else:
            converted[new_key] = val

    return converted


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
        import importlib.metadata

        import diffusers.quantizers.gguf.gguf_quantizer as _gguf_quantizer_mod
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            try:
                _gguf_ver = importlib.metadata.version("gguf")
                _diffusers_import_utils._gguf_available = True
                _diffusers_import_utils._gguf_version = _gguf_ver
                from diffusers.quantizers.gguf.utils import (
                    GGML_QUANT_SIZES,
                    GGUFParameter,
                    _dequantize_gguf_and_restore_linear,
                    _quant_shape_from_byte_shape,
                    _replace_with_gguf_linear,
                )

                _gguf_quantizer_mod.GGML_QUANT_SIZES = GGML_QUANT_SIZES
                _gguf_quantizer_mod.GGUFParameter = GGUFParameter
                _gguf_quantizer_mod._dequantize_gguf_and_restore_linear = (
                    _dequantize_gguf_and_restore_linear
                )
                _gguf_quantizer_mod._quant_shape_from_byte_shape = (
                    _quant_shape_from_byte_shape
                )
                _gguf_quantizer_mod._replace_with_gguf_linear = (
                    _replace_with_gguf_linear
                )
            except importlib.metadata.PackageNotFoundError:
                pass

        import accelerate.big_modeling as _accel_bm
        import diffusers.loaders.single_file_model as _sfm
        import torch.nn as nn
        from diffusers import GGUFQuantizationConfig, HunyuanVideo15Transformer3DModel
        from huggingface_hub import hf_hub_download

        # HunyuanVideo15Transformer3DModel is not in diffusers' SINGLE_FILE_LOADABLE_CLASSES;
        # register it with a custom key converter so from_single_file works.
        if "HunyuanVideo15Transformer3DModel" not in _sfm.SINGLE_FILE_LOADABLE_CLASSES:
            _sfm.SINGLE_FILE_LOADABLE_CLASSES["HunyuanVideo15Transformer3DModel"] = {
                "checkpoint_mapping_fn": _convert_hunyuan_video_1_5_i2v_transformer_to_diffusers,
                "default_subfolder": "transformer",
            }

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        # Local config with correct HunyuanVideo15Transformer3DModel dimensions.
        # Without this, from_single_file fetches hunyuanvideo-community/HunyuanVideo
        # config (24 heads, 3072 dim) which mismatches the 1.5 checkpoint.
        local_config_dir = os.path.join(os.path.dirname(__file__), "model_config")

        orig_dispatch = _accel_bm.dispatch_model

        def _patched_dispatch(model, device_map=None, **kwargs):
            if device_map:
                for name, param in list(model.named_parameters()):
                    if param.device.type != "meta":
                        continue
                    parts = name.split(".")
                    module = model
                    for part in parts[:-1]:
                        module = getattr(module, part)
                    module._parameters[parts[-1]] = nn.Parameter(
                        torch.zeros(param.shape, dtype=compute_dtype, device="cpu"),
                        requires_grad=param.requires_grad,
                    )
            return orig_dispatch(model, device_map=device_map, **kwargs)

        _accel_bm.dispatch_model = _patched_dispatch
        _sfm.dispatch_model = _patched_dispatch
        try:
            self._transformer = HunyuanVideo15Transformer3DModel.from_single_file(
                gguf_path,
                config=local_config_dir,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            _accel_bm.dispatch_model = orig_dispatch
            _sfm.dispatch_model = orig_dispatch

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
        timestep = torch.tensor([500.0], dtype=dtype).expand(batch_size)

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
