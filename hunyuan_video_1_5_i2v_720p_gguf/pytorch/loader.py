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
BASE_CONFIG_REPO = "tencent/HunyuanVideo-1.5"

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

_CFG_DISTILLED_VARIANTS = {
    ModelVariant.I2V_720P_CFG_DISTILLED_Q4_K_S,
    ModelVariant.I2V_720P_CFG_DISTILLED_Q4_K_M,
    ModelVariant.I2V_720P_CFG_DISTILLED_Q5_K_S,
    ModelVariant.I2V_720P_CFG_DISTILLED_Q5_K_M,
    ModelVariant.I2V_720P_CFG_DISTILLED_Q6_K,
    ModelVariant.I2V_720P_CFG_DISTILLED_Q8_0,
}


def _convert_hunyuan_video15_gguf_to_diffusers(checkpoint, **kwargs):
    """Convert jayn7/HunyuanVideo-1.5_I2V_720p-GGUF keys to diffusers format.

    The GGUF file uses HunyuanVideo's original key naming convention. This
    function remaps them to the HunyuanVideo15Transformer3DModel parameter
    names expected by diffusers.
    """
    state_dict = dict(checkpoint)
    new_state_dict = {}

    for key in list(state_dict.keys()):
        value = state_dict[key]

        # ── img_in → x_embedder ──────────────────────────────────────────────
        if key.startswith("img_in."):
            new_key = key.replace("img_in.", "x_embedder.", 1)
            new_state_dict[new_key] = value

        # ── cond_type_embedding → cond_type_embed ───────────────────────────
        elif key == "cond_type_embedding.weight":
            new_state_dict["cond_type_embed.weight"] = value

        # ── time_in → time_embed ─────────────────────────────────────────────
        elif key.startswith("time_in.mlp.0."):
            suffix = key[len("time_in.mlp.0."):]
            new_state_dict[f"time_embed.timestep_embedder.linear_1.{suffix}"] = value
        elif key.startswith("time_in.mlp.2."):
            suffix = key[len("time_in.mlp.2."):]
            new_state_dict[f"time_embed.timestep_embedder.linear_2.{suffix}"] = value

        # ── vision_in → image_embedder (Sequential 0,1,3,4 → norm_in,linear_1,linear_2,norm_out) ──
        elif key.startswith("vision_in.proj.0."):
            suffix = key[len("vision_in.proj.0."):]
            new_state_dict[f"image_embedder.norm_in.{suffix}"] = value
        elif key.startswith("vision_in.proj.1."):
            suffix = key[len("vision_in.proj.1."):]
            new_state_dict[f"image_embedder.linear_1.{suffix}"] = value
        elif key.startswith("vision_in.proj.3."):
            suffix = key[len("vision_in.proj.3."):]
            new_state_dict[f"image_embedder.linear_2.{suffix}"] = value
        elif key.startswith("vision_in.proj.4."):
            suffix = key[len("vision_in.proj.4."):]
            new_state_dict[f"image_embedder.norm_out.{suffix}"] = value

        # ── byt5_in → context_embedder_2 ─────────────────────────────────────
        elif key.startswith("byt5_in.fc1."):
            suffix = key[len("byt5_in.fc1."):]
            new_state_dict[f"context_embedder_2.linear_1.{suffix}"] = value
        elif key.startswith("byt5_in.fc2."):
            suffix = key[len("byt5_in.fc2."):]
            new_state_dict[f"context_embedder_2.linear_2.{suffix}"] = value
        elif key.startswith("byt5_in.fc3."):
            suffix = key[len("byt5_in.fc3."):]
            new_state_dict[f"context_embedder_2.linear_3.{suffix}"] = value
        elif key.startswith("byt5_in.layernorm."):
            suffix = key[len("byt5_in.layernorm."):]
            new_state_dict[f"context_embedder_2.norm.{suffix}"] = value

        # ── txt_in → context_embedder ─────────────────────────────────────────
        elif key.startswith("txt_in.input_embedder."):
            suffix = key[len("txt_in.input_embedder."):]
            new_state_dict[f"context_embedder.proj_in.{suffix}"] = value
        elif key.startswith("txt_in.c_embedder.linear_1."):
            suffix = key[len("txt_in.c_embedder.linear_1."):]
            new_state_dict[f"context_embedder.time_text_embed.text_embedder.linear_1.{suffix}"] = value
        elif key.startswith("txt_in.c_embedder.linear_2."):
            suffix = key[len("txt_in.c_embedder.linear_2."):]
            new_state_dict[f"context_embedder.time_text_embed.text_embedder.linear_2.{suffix}"] = value
        elif key.startswith("txt_in.t_embedder.mlp.0."):
            suffix = key[len("txt_in.t_embedder.mlp.0."):]
            new_state_dict[f"context_embedder.time_text_embed.timestep_embedder.linear_1.{suffix}"] = value
        elif key.startswith("txt_in.t_embedder.mlp.2."):
            suffix = key[len("txt_in.t_embedder.mlp.2."):]
            new_state_dict[f"context_embedder.time_text_embed.timestep_embedder.linear_2.{suffix}"] = value
        elif key.startswith("txt_in.individual_token_refiner.blocks."):
            rest = key[len("txt_in.individual_token_refiner.blocks."):]
            # rest is "N.subkey.suffix"
            dot = rest.index(".")
            block_idx = rest[:dot]
            subkey = rest[dot + 1:]
            prefix = f"context_embedder.token_refiner.refiner_blocks.{block_idx}"

            if subkey.startswith("self_attn_qkv."):
                suffix = subkey[len("self_attn_qkv."):]
                tensor = value
                q, k, v = tensor.chunk(3, dim=0)
                new_state_dict[f"{prefix}.attn.to_q.{suffix}"] = q
                new_state_dict[f"{prefix}.attn.to_k.{suffix}"] = k
                new_state_dict[f"{prefix}.attn.to_v.{suffix}"] = v
            elif subkey.startswith("self_attn_proj."):
                suffix = subkey[len("self_attn_proj."):]
                new_state_dict[f"{prefix}.attn.to_out.0.{suffix}"] = value
            elif subkey.startswith("mlp.fc1."):
                suffix = subkey[len("mlp.fc1."):]
                new_state_dict[f"{prefix}.ff.net.0.proj.{suffix}"] = value
            elif subkey.startswith("mlp.fc2."):
                suffix = subkey[len("mlp.fc2."):]
                new_state_dict[f"{prefix}.ff.net.2.{suffix}"] = value
            elif subkey.startswith("adaLN_modulation.1."):
                suffix = subkey[len("adaLN_modulation.1."):]
                new_state_dict[f"{prefix}.norm_out.linear.{suffix}"] = value
            else:
                new_state_dict[f"{prefix}.{subkey}"] = value

        # ── double_blocks → transformer_blocks ───────────────────────────────
        elif key.startswith("double_blocks."):
            rest = key[len("double_blocks."):]
            dot = rest.index(".")
            block_idx = rest[:dot]
            subkey = rest[dot + 1:]
            prefix = f"transformer_blocks.{block_idx}"

            if subkey.startswith("img_attn_qkv."):
                suffix = subkey[len("img_attn_qkv."):]
                q, k, v = value.chunk(3, dim=0)
                new_state_dict[f"{prefix}.attn.to_q.{suffix}"] = q
                new_state_dict[f"{prefix}.attn.to_k.{suffix}"] = k
                new_state_dict[f"{prefix}.attn.to_v.{suffix}"] = v
            elif subkey.startswith("txt_attn_qkv."):
                suffix = subkey[len("txt_attn_qkv."):]
                q, k, v = value.chunk(3, dim=0)
                new_state_dict[f"{prefix}.attn.add_q_proj.{suffix}"] = q
                new_state_dict[f"{prefix}.attn.add_k_proj.{suffix}"] = k
                new_state_dict[f"{prefix}.attn.add_v_proj.{suffix}"] = v
            elif subkey.startswith("img_attn_proj."):
                suffix = subkey[len("img_attn_proj."):]
                new_state_dict[f"{prefix}.attn.to_out.0.{suffix}"] = value
            elif subkey.startswith("txt_attn_proj."):
                suffix = subkey[len("txt_attn_proj."):]
                new_state_dict[f"{prefix}.attn.to_add_out.{suffix}"] = value
            elif subkey == "img_attn_q_norm.weight":
                new_state_dict[f"{prefix}.attn.norm_q.weight"] = value
            elif subkey == "img_attn_k_norm.weight":
                new_state_dict[f"{prefix}.attn.norm_k.weight"] = value
            elif subkey == "txt_attn_q_norm.weight":
                new_state_dict[f"{prefix}.attn.norm_added_q.weight"] = value
            elif subkey == "txt_attn_k_norm.weight":
                new_state_dict[f"{prefix}.attn.norm_added_k.weight"] = value
            elif subkey.startswith("img_mlp.fc1."):
                suffix = subkey[len("img_mlp.fc1."):]
                new_state_dict[f"{prefix}.ff.net.0.proj.{suffix}"] = value
            elif subkey.startswith("img_mlp.fc2."):
                suffix = subkey[len("img_mlp.fc2."):]
                new_state_dict[f"{prefix}.ff.net.2.{suffix}"] = value
            elif subkey.startswith("txt_mlp.fc1."):
                suffix = subkey[len("txt_mlp.fc1."):]
                new_state_dict[f"{prefix}.ff_context.net.0.proj.{suffix}"] = value
            elif subkey.startswith("txt_mlp.fc2."):
                suffix = subkey[len("txt_mlp.fc2."):]
                new_state_dict[f"{prefix}.ff_context.net.2.{suffix}"] = value
            elif subkey.startswith("img_mod.linear."):
                suffix = subkey[len("img_mod.linear."):]
                new_state_dict[f"{prefix}.norm1.linear.{suffix}"] = value
            elif subkey.startswith("txt_mod.linear."):
                suffix = subkey[len("txt_mod.linear."):]
                new_state_dict[f"{prefix}.norm1_context.linear.{suffix}"] = value
            else:
                new_state_dict[f"{prefix}.{subkey}"] = value

        # ── final_layer → norm_out + proj_out ────────────────────────────────
        elif key.startswith("final_layer.adaLN_modulation.1."):
            suffix = key[len("final_layer.adaLN_modulation.1."):]
            new_state_dict[f"norm_out.linear.{suffix}"] = value
        elif key.startswith("final_layer.linear."):
            suffix = key[len("final_layer.linear."):]
            new_state_dict[f"proj_out.{suffix}"] = value

        else:
            new_state_dict[key] = value

    return new_state_dict


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
        from diffusers.quantizers.gguf.utils import GGUFParameter

        # GGUFParameter.as_tensor() calls torch.Tensor._make_subclass() which
        # re-enters GGUFParameter.__torch_function__ → super().__torch_function__
        # recursively when traced by dynamo. Patching with DisableTorchFunctionSubclass
        # escapes the subclass so the returned tensor is a plain torch.Tensor.
        def _safe_as_tensor(self):
            with torch._C.DisableTorchFunctionSubclass():
                return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)

        GGUFParameter.as_tensor = _safe_as_tensor

        # Register HunyuanVideo15Transformer3DModel in SINGLE_FILE_LOADABLE_CLASSES
        # with the key conversion function that maps jayn7 GGUF keys to diffusers.
        if "HunyuanVideo15Transformer3DModel" not in SINGLE_FILE_LOADABLE_CLASSES:
            SINGLE_FILE_LOADABLE_CLASSES["HunyuanVideo15Transformer3DModel"] = {
                "checkpoint_mapping_fn": _convert_hunyuan_video15_gguf_to_diffusers,
            }

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        config_subfolder = (
            "transformer/720p_i2v_distilled"
            if self._variant in _CFG_DISTILLED_VARIANTS
            else "transformer/720p_i2v"
        )

        # The tencent/HunyuanVideo-1.5 config JSON was saved with an older
        # diffusers API and has three mismatches vs the current class:
        #   1. patch_size stored as list [t, h, w] and patch_size_t as null;
        #      __init__ expects separate int scalars.
        #   2. qk_norm stored as bool (true) with qk_norm_type "rms"; __init__
        #      expects a string like "rms_norm".
        #   3. in_channels=32 in config but actual i2v model uses 65 input
        #      channels (video latent + image conditioning). The class default
        #      is also 65; the config value is simply wrong.
        _orig_from_config = HunyuanVideo15Transformer3DModel.from_config.__func__

        @classmethod  # type: ignore[misc]
        def _patched_from_config(cls, config, **kwargs):
            if isinstance(config, dict):
                config = dict(config)
                if isinstance(config.get("patch_size"), list):
                    ps_list = config["patch_size"]
                    if config.get("patch_size_t") is None and len(ps_list) >= 1:
                        config["patch_size_t"] = int(ps_list[0])
                    config["patch_size"] = int(ps_list[1]) if len(ps_list) > 1 else int(ps_list[0])
                if isinstance(config.get("qk_norm"), bool) and config["qk_norm"]:
                    norm_type = config.get("qk_norm_type", "rms")
                    if not norm_type.endswith("_norm"):
                        norm_type = norm_type + "_norm"
                    config["qk_norm"] = norm_type
                if config.get("in_channels") == 32:
                    config["in_channels"] = 65
            return _orig_from_config(cls, config, **kwargs)

        HunyuanVideo15Transformer3DModel.from_config = _patched_from_config

        try:
            self._transformer = HunyuanVideo15Transformer3DModel.from_single_file(
                f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
                config=BASE_CONFIG_REPO,
                subfolder=config_subfolder,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            HunyuanVideo15Transformer3DModel.from_config = classmethod(_orig_from_config)

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
        timestep = torch.tensor([500], dtype=torch.bfloat16).expand(batch_size)

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
