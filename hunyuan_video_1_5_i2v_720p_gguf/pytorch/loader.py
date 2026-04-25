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


def _convert_hunyuan_video15_checkpoint(checkpoint):
    """Convert HunyuanVideo 1.5 GGUF checkpoint keys to diffusers format.

    The GGUF file stores weights under the original model's key naming
    convention (e.g. double_blocks, img_in, vision_in, byt5_in). This
    function renames them to match HunyuanVideo15Transformer3DModel's
    state dict layout.
    """

    def remap_norm_scale_shift_(key, state_dict):
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        state_dict[
            key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")
        ] = new_weight

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

    # Simple key renames applied in the first pass.
    # Note: vision_in.proj.N renames must come before the generic img_in rename
    # to avoid partial matches. The more specific patterns are listed first so
    # sequential str.replace calls don't clobber them.
    KEYS_RENAME = {
        # image patch & vision embedders
        "img_in": "x_embedder",
        "vision_in.proj.0": "image_embedder.norm_in",
        "vision_in.proj.1": "image_embedder.linear_1",
        "vision_in.proj.3": "image_embedder.linear_2",
        "vision_in.proj.4": "image_embedder.norm_out",
        # byt5 text projection (context_embedder_2)
        "byt5_in.layernorm": "context_embedder_2.norm",
        "byt5_in.fc1": "context_embedder_2.linear_1",
        "byt5_in.fc2": "context_embedder_2.linear_2",
        "byt5_in.fc3": "context_embedder_2.linear_3",
        # condition type embedding
        "cond_type_embedding": "cond_type_embed",
        # time embedding
        "time_in.mlp.0": "time_embed.timestep_embedder.linear_1",
        "time_in.mlp.2": "time_embed.timestep_embedder.linear_2",
        # transformer blocks
        "double_blocks": "transformer_blocks",
        "img_attn_q_norm": "attn.norm_q",
        "img_attn_k_norm": "attn.norm_k",
        "img_attn_proj": "attn.to_out.0",
        "txt_attn_q_norm": "attn.norm_added_q",
        "txt_attn_k_norm": "attn.norm_added_k",
        "txt_attn_proj": "attn.to_add_out",
        "img_mod.linear": "norm1.linear",
        "img_mlp.fc1": "ff.net.0.proj",
        "img_mlp.fc2": "ff.net.2",
        "txt_mod.linear": "norm1_context.linear",
        "txt_mlp.fc1": "ff_context.net.0.proj",
        "txt_mlp.fc2": "ff_context.net.2",
        # final layer
        "final_layer.linear": "proj_out",
        # token refiner helpers (applied before remap_txt_in_ sees the key)
        "self_attn_proj": "attn.to_out.0",
        "input_embedder": "proj_in",
        "fc1": "net.0.proj",
        "fc2": "net.2",
    }

    # Keys handled by special remapping functions (run in second pass).
    SPECIAL_KEYS = {
        "txt_in": remap_txt_in_,
        "img_attn_qkv": remap_img_attn_qkv_,
        "txt_attn_qkv": remap_txt_attn_qkv_,
        "final_layer.adaLN_modulation.1": remap_norm_scale_shift_,
    }

    # First pass: simple string replacements.
    for key in list(checkpoint.keys()):
        new_key = key
        for old, new in KEYS_RENAME.items():
            new_key = new_key.replace(old, new)
        if new_key != key:
            checkpoint[new_key] = checkpoint.pop(key)

    # Second pass: special-case handlers.
    for key in list(checkpoint.keys()):
        for special_key, handler in SPECIAL_KEYS.items():
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

        diffusers 0.37.1 does not include HunyuanVideo15Transformer3DModel in
        the from_single_file whitelist, so we replicate that loading flow here:
        download the GGUF, convert key names to diffusers format, then load via
        the GGUFQuantizer.
        """
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from accelerate import init_empty_weights
        from diffusers import GGUFQuantizationConfig, HunyuanVideo15Transformer3DModel
        from diffusers.loaders.single_file_utils import load_single_file_checkpoint
        from diffusers.models.model_loading_utils import load_model_dict_into_meta
        from diffusers.quantizers.auto import DiffusersAutoQuantizer

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]
        gguf_url = f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}"

        # Download and parse the GGUF checkpoint.
        checkpoint = load_single_file_checkpoint(gguf_url)

        # Set up GGUF quantizer.
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        hf_quantizer = DiffusersAutoQuantizer.from_config(quantization_config)
        hf_quantizer.validate_environment()
        torch_dtype = hf_quantizer.update_torch_dtype(compute_dtype)

        # Convert GGUF key names to diffusers format.
        checkpoint = _convert_hunyuan_video15_checkpoint(checkpoint)

        # Instantiate the model skeleton on the meta device.
        with init_empty_weights():
            model = HunyuanVideo15Transformer3DModel()

        # Replace linear layers with GGUF-quantized equivalents.
        hf_quantizer.preprocess_model(
            model=model,
            device_map=None,
            state_dict=checkpoint,
            keep_in_fp32_modules=[],
        )

        # Identify unexpected keys so load_model_dict_into_meta can warn about them.
        model_state_dict = model.state_dict()
        unexpected_keys = [k for k in checkpoint if k not in model_state_dict]

        # Load the quantized weights into the model.
        load_model_dict_into_meta(
            model,
            checkpoint,
            dtype=torch_dtype,
            device_map={"": torch.device("cpu")},
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=[],
            unexpected_keys=unexpected_keys,
        )

        hf_quantizer.postprocess_model(model)
        model.hf_quantizer = hf_quantizer
        model.eval()

        self._transformer = model
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
