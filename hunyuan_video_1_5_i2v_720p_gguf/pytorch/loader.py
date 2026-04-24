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


def _convert_hunyuan_video15_i2v_gguf_to_diffusers(checkpoint, **kwargs):
    """Convert HunyuanVideo 1.5 I2V GGUF checkpoint keys to diffusers format."""
    import torch

    def remap_norm_scale_shift_(key, state_dict):
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        state_dict[
            key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")
        ] = torch.cat([scale, shift], dim=0)

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

    # More specific renames must come before general ones to prevent partial matches.
    RENAME_DICT = {
        "img_in.proj": "x_embedder.proj",
        "vision_in.proj.0": "image_embedder.norm_in",
        "vision_in.proj.1": "image_embedder.linear_1",
        "vision_in.proj.3": "image_embedder.linear_2",
        "vision_in.proj.4": "image_embedder.norm_out",
        "byt5_in.layernorm": "context_embedder_2.norm",
        "byt5_in.fc1": "context_embedder_2.linear_1",
        "byt5_in.fc2": "context_embedder_2.linear_2",
        "byt5_in.fc3": "context_embedder_2.linear_3",
        "time_in.mlp.0": "time_embed.timestep_embedder.linear_1",
        "time_in.mlp.2": "time_embed.timestep_embedder.linear_2",
        "cond_type_embedding": "cond_type_embed",
        "double_blocks": "transformer_blocks",
        "img_attn_q_norm": "attn.norm_q",
        "img_attn_k_norm": "attn.norm_k",
        "img_attn_proj": "attn.to_out.0",
        "txt_attn_q_norm": "attn.norm_added_q",
        "txt_attn_k_norm": "attn.norm_added_k",
        "txt_attn_proj": "attn.to_add_out",
        "img_mod.linear": "norm1.linear",
        "img_mlp": "ff",
        "txt_mod.linear": "norm1_context.linear",
        "txt_mlp": "ff_context",
        "self_attn_proj": "attn.to_out.0",
        "final_layer.linear": "proj_out",
        "input_embedder": "proj_in",
        "fc1": "net.0.proj",
        "fc2": "net.2",
    }

    SPECIAL_KEYS = {
        "txt_in": remap_txt_in_,
        "img_attn_qkv": remap_img_attn_qkv_,
        "txt_attn_qkv": remap_txt_attn_qkv_,
        "final_layer.adaLN_modulation.1": remap_norm_scale_shift_,
    }

    for key in list(checkpoint.keys()):
        new_key = key
        for old, new in RENAME_DICT.items():
            new_key = new_key.replace(old, new)
        if new_key != key:
            checkpoint[new_key] = checkpoint.pop(key)

    for key in list(checkpoint.keys()):
        for special_key, handler_fn in SPECIAL_KEYS.items():
            if special_key in key:
                handler_fn(key, checkpoint)
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

        Downloads the GGUF checkpoint, converts keys to diffusers format, and
        loads quantized weights into a HunyuanVideo15Transformer3DModel with
        the correct I2V architecture config.
        """
        import importlib
        import importlib.metadata
        import importlib.util

        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        if _diffusers_import_utils._gguf_version == "N/A":
            try:
                _diffusers_import_utils._gguf_version = importlib.metadata.version(
                    "gguf"
                )
            except importlib.metadata.PackageNotFoundError:
                pass

        # If the diffusers GGUF quantizer was loaded before gguf was installed,
        # its module-level conditional imports (e.g. _replace_with_gguf_linear)
        # were skipped.  Patch the quantizer module globals directly so the
        # already-imported GGUFQuantizer class can find the symbols it needs.
        if _diffusers_import_utils._gguf_available:
            import diffusers.quantizers.gguf.gguf_quantizer as _gguf_quantizer_mod

            if not hasattr(_gguf_quantizer_mod, "_replace_with_gguf_linear"):
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

        from accelerate import init_empty_weights
        from diffusers import GGUFQuantizationConfig, HunyuanVideo15Transformer3DModel
        from diffusers.models.model_loading_utils import (
            load_gguf_checkpoint,
            load_model_dict_into_meta,
        )
        from diffusers.quantizers import DiffusersAutoQuantizer
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        # Load GGUF checkpoint and convert keys to diffusers format.
        checkpoint = load_gguf_checkpoint(gguf_path)
        diffusers_checkpoint = _convert_hunyuan_video15_i2v_gguf_to_diffusers(
            checkpoint
        )

        # Create the model with its default I2V architecture config on meta device.
        with init_empty_weights():
            model = HunyuanVideo15Transformer3DModel()

        # Set up GGUF quantizer and replace nn.Linear layers with GGUFLinear.
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        hf_quantizer = DiffusersAutoQuantizer.from_config(quantization_config)
        hf_quantizer.validate_environment()
        hf_quantizer.preprocess_model(
            model=model,
            device_map=None,
            state_dict=diffusers_checkpoint,
            keep_in_fp32_modules=[],
        )

        # Populate meta tensors from the GGUF checkpoint.
        load_model_dict_into_meta(
            model,
            diffusers_checkpoint,
            dtype=compute_dtype,
            device_map={"": torch.device("cpu")},
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=[],
            unexpected_keys=[],
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
        # HunyuanVideo15TimestepEmbedder casts sinusoidal projections back to
        # timestep.dtype; passing Long would produce Long→Float dtype mismatch.
        timestep = torch.tensor([500], dtype=dtype).expand(batch_size)

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
