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

# Diffusers repo that provides the transformer architecture config.
# The 480p and 720p I2V models share the same HunyuanVideo15Transformer3DModel
# architecture so this config is valid for both resolutions.
_CONFIG_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"

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


def _convert_hunyuan_video15_i2v_gguf_to_diffusers(checkpoint, **kwargs):
    """Convert HunyuanVideo 1.5 I2V GGUF checkpoint from original to diffusers key format.

    The GGUF files in jayn7/HunyuanVideo-1.5_I2V_720p-GGUF use the original
    ComfyUI/upstream key naming convention. This function remaps those keys to
    the diffusers HunyuanVideo15Transformer3DModel state-dict layout so that
    from_single_file can load the weights correctly.
    """

    def remap_norm_scale_shift_(key, state_dict):
        # final_layer.adaLN_modulation.1 stores [shift, scale]; diffusers expects
        # [scale, shift] for norm_out.linear.
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        new_key = key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")
        state_dict[new_key] = torch.cat([scale, shift], dim=0)

    def remap_txt_in_(key, state_dict):
        def rename_key(k):
            k = k.replace(
                "individual_token_refiner.blocks", "token_refiner.refiner_blocks"
            )
            k = k.replace("adaLN_modulation.1", "norm_out.linear")
            k = k.replace("txt_in", "context_embedder")
            k = k.replace(
                "t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1"
            )
            k = k.replace(
                "t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2"
            )
            k = k.replace("c_embedder", "time_text_embed.text_embedder")
            k = k.replace("input_embedder", "proj_in")
            k = k.replace("mlp", "ff")
            return k

        if "self_attn_qkv" in key:
            weight = state_dict.pop(key)
            to_q, to_k, to_v = weight.chunk(3, dim=0)
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_q"))] = to_q
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_k"))] = to_k
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_v"))] = to_v
        else:
            state_dict[rename_key(key)] = state_dict.pop(key)

    def remap_img_attn_qkv_(key, state_dict):
        weight = state_dict.pop(key)
        to_q, to_k, to_v = weight.chunk(3, dim=0)
        state_dict[key.replace("img_attn_qkv", "attn.to_q")] = to_q
        state_dict[key.replace("img_attn_qkv", "attn.to_k")] = to_k
        state_dict[key.replace("img_attn_qkv", "attn.to_v")] = to_v

    def remap_txt_attn_qkv_(key, state_dict):
        weight = state_dict.pop(key)
        to_q, to_k, to_v = weight.chunk(3, dim=0)
        state_dict[key.replace("txt_attn_qkv", "attn.add_q_proj")] = to_q
        state_dict[key.replace("txt_attn_qkv", "attn.add_k_proj")] = to_k
        state_dict[key.replace("txt_attn_qkv", "attn.add_v_proj")] = to_v

    def remap_vision_in_(key, state_dict):
        # vision_in.proj is a Sequential: [Linear(0), Norm(1), Act(2), Linear(3), Norm(4)]
        # Index 2 has no parameters.  Map the four parameterised layers to the
        # named attributes of diffusers' image_embedder.
        _MAP = {
            "vision_in.proj.0": "image_embedder.norm_in",
            "vision_in.proj.1": "image_embedder.linear_1",
            "vision_in.proj.3": "image_embedder.linear_2",
            "vision_in.proj.4": "image_embedder.norm_out",
        }
        for old_prefix, new_prefix in _MAP.items():
            if key.startswith(old_prefix + "."):
                state_dict[key.replace(old_prefix, new_prefix, 1)] = state_dict.pop(key)
                return

    # Simple key renames applied in one pass.  Order matters: more specific
    # patterns must appear before generic ones that are substrings of them.
    RENAME_DICT = {
        "cond_type_embedding": "cond_type_embed",
        # byt5_in must come before the generic fc1/fc2/fc3 patterns.
        "byt5_in.fc1": "context_embedder_2.linear_1",
        "byt5_in.fc2": "context_embedder_2.linear_2",
        "byt5_in.fc3": "context_embedder_2.linear_3",
        "byt5_in.layernorm": "context_embedder_2.norm",
        "img_in.proj": "x_embedder.proj",
        # time_in.mlp.N must come before the generic mlp/fc patterns.
        "time_in.mlp.0": "time_embed.timestep_embedder.linear_1",
        "time_in.mlp.2": "time_embed.timestep_embedder.linear_2",
        "double_blocks": "transformer_blocks",
        "img_attn_q_norm": "attn.norm_q",
        "img_attn_k_norm": "attn.norm_k",
        "img_attn_proj": "attn.to_out.0",
        "txt_attn_q_norm": "attn.norm_added_q",
        "txt_attn_k_norm": "attn.norm_added_k",
        "txt_attn_proj": "attn.to_add_out",
        "img_mod.linear": "norm1.linear",
        # img_mlp/txt_mlp must come before the generic fc1/fc2 patterns.
        "img_mlp": "ff",
        "txt_mod.linear": "norm1_context.linear",
        "txt_mlp": "ff_context",
        "self_attn_proj": "attn.to_out.0",
        "fc1": "net.0.proj",
        "fc2": "net.2",
        "final_layer.linear": "proj_out",
    }

    for key in list(checkpoint.keys()):
        new_key = key
        for old, new in RENAME_DICT.items():
            new_key = new_key.replace(old, new)
        if new_key != key:
            checkpoint[new_key] = checkpoint.pop(key)

    SPECIAL_KEYS_REMAP = {
        "txt_in": remap_txt_in_,
        "img_attn_qkv": remap_img_attn_qkv_,
        "txt_attn_qkv": remap_txt_attn_qkv_,
        "vision_in": remap_vision_in_,
        "final_layer.adaLN_modulation.1": remap_norm_scale_shift_,
    }

    for key in list(checkpoint.keys()):
        for special_key, handler in SPECIAL_KEYS_REMAP.items():
            if special_key in key:
                handler(key, checkpoint)
                break

    return checkpoint


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

        # diffusers 0.37.x does not include HunyuanVideo15Transformer3DModel in
        # SINGLE_FILE_LOADABLE_CLASSES, so from_single_file raises ValueError.
        # Register it here with the correct conversion function that maps the
        # original ComfyUI-style GGUF key names to diffusers format.
        if "HunyuanVideo15Transformer3DModel" not in SINGLE_FILE_LOADABLE_CLASSES:
            SINGLE_FILE_LOADABLE_CLASSES["HunyuanVideo15Transformer3DModel"] = {
                "checkpoint_mapping_fn": _convert_hunyuan_video15_i2v_gguf_to_diffusers,
                "default_subfolder": "transformer",
            }

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        # Provide the transformer config explicitly so from_single_file does not
        # fall back to auto-detecting the model type from checkpoint keys (the
        # GGUF uses original key names that are not recognised by
        # infer_diffusers_model_type, which would default to a wrong SD v1 config).
        # URL must omit "resolve/main/" so _extract_repo_id_and_weights_name
        # correctly separates the repo id from the nested sub-path.
        self._transformer = HunyuanVideo15Transformer3DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/{gguf_file}",
            config=_CONFIG_REPO,
            subfolder="transformer",
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
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

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
