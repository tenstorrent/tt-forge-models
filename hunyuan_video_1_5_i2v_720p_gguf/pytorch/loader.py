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


def _convert_gguf_checkpoint_to_diffusers(checkpoint):
    """Convert Tencent GGUF key names to diffusers HunyuanVideo15Transformer3DModel names.

    The GGUF files from jayn7/HunyuanVideo-1.5_I2V_720p-GGUF use the original
    Tencent checkpoint naming.  This function renames every key to match the
    diffusers layout so the resulting dict can be loaded with
    load_model_dict_into_meta.

    GGUFParameter.__torch_function__ preserves the subclass (and quant_type)
    through tensor operations, so split QKV tensors remain GGUFParameter.
    """
    new = {}

    for key, value in checkpoint.items():
        # ------------------------------------------------------------------ #
        # x_embedder  (img_in.proj → x_embedder.proj)
        # ------------------------------------------------------------------ #
        if key.startswith("img_in."):
            new[key.replace("img_in.", "x_embedder.")] = value

        # ------------------------------------------------------------------ #
        # image_embedder  (vision_in.proj.{0,1,3,4} → image_embedder.*)
        # Sequential order: norm_in(0), linear_1(1), GELU(2), linear_2(3), norm_out(4)
        # ------------------------------------------------------------------ #
        elif key.startswith("vision_in.proj.0."):
            new[key.replace("vision_in.proj.0.", "image_embedder.norm_in.")] = value
        elif key.startswith("vision_in.proj.1."):
            new[key.replace("vision_in.proj.1.", "image_embedder.linear_1.")] = value
        elif key.startswith("vision_in.proj.3."):
            new[key.replace("vision_in.proj.3.", "image_embedder.linear_2.")] = value
        elif key.startswith("vision_in.proj.4."):
            new[key.replace("vision_in.proj.4.", "image_embedder.norm_out.")] = value

        # ------------------------------------------------------------------ #
        # context_embedder_2  (byt5_in → context_embedder_2)
        # ------------------------------------------------------------------ #
        elif key.startswith("byt5_in.layernorm."):
            new[key.replace("byt5_in.layernorm.", "context_embedder_2.norm.")] = value
        elif key.startswith("byt5_in.fc1."):
            new[key.replace("byt5_in.fc1.", "context_embedder_2.linear_1.")] = value
        elif key.startswith("byt5_in.fc2."):
            new[key.replace("byt5_in.fc2.", "context_embedder_2.linear_2.")] = value
        elif key.startswith("byt5_in.fc3."):
            new[key.replace("byt5_in.fc3.", "context_embedder_2.linear_3.")] = value

        # ------------------------------------------------------------------ #
        # time_embed  (time_in.mlp → time_embed.timestep_embedder)
        # ------------------------------------------------------------------ #
        elif key.startswith("time_in.mlp.0."):
            new[
                key.replace("time_in.mlp.0.", "time_embed.timestep_embedder.linear_1.")
            ] = value
        elif key.startswith("time_in.mlp.2."):
            new[
                key.replace("time_in.mlp.2.", "time_embed.timestep_embedder.linear_2.")
            ] = value

        # ------------------------------------------------------------------ #
        # cond_type_embed
        # ------------------------------------------------------------------ #
        elif key == "cond_type_embedding.weight":
            new["cond_type_embed.weight"] = value

        # ------------------------------------------------------------------ #
        # context_embedder  (txt_in → context_embedder)
        # ------------------------------------------------------------------ #
        elif key.startswith("txt_in.input_embedder."):
            new[
                key.replace("txt_in.input_embedder.", "context_embedder.proj_in.")
            ] = value
        elif key.startswith("txt_in.t_embedder.mlp.0."):
            new[
                key.replace(
                    "txt_in.t_embedder.mlp.0.",
                    "context_embedder.time_text_embed.timestep_embedder.linear_1.",
                )
            ] = value
        elif key.startswith("txt_in.t_embedder.mlp.2."):
            new[
                key.replace(
                    "txt_in.t_embedder.mlp.2.",
                    "context_embedder.time_text_embed.timestep_embedder.linear_2.",
                )
            ] = value
        elif key.startswith("txt_in.c_embedder.linear_1."):
            new[
                key.replace(
                    "txt_in.c_embedder.linear_1.",
                    "context_embedder.time_text_embed.text_embedder.linear_1.",
                )
            ] = value
        elif key.startswith("txt_in.c_embedder.linear_2."):
            new[
                key.replace(
                    "txt_in.c_embedder.linear_2.",
                    "context_embedder.time_text_embed.text_embedder.linear_2.",
                )
            ] = value

        # Individual token refiner blocks
        elif "txt_in.individual_token_refiner.blocks." in key:
            # txt_in.individual_token_refiner.blocks.N.* →
            #   context_embedder.token_refiner.refiner_blocks.N.*
            k = key.replace(
                "txt_in.individual_token_refiner.blocks.",
                "context_embedder.token_refiner.refiner_blocks.",
            )
            if "self_attn_qkv" in k:
                # Split [3*H, D] → to_q [H,D], to_k [H,D], to_v [H,D]
                to_q, to_k, to_v = value.chunk(3, dim=0)
                suffix = k.split("self_attn_qkv")[1]  # ".weight" or ".bias"
                base = k.split("self_attn_qkv")[0]
                new[base + "attn.to_q" + suffix] = to_q
                new[base + "attn.to_k" + suffix] = to_k
                new[base + "attn.to_v" + suffix] = to_v
            elif "self_attn_proj." in k:
                new[k.replace("self_attn_proj.", "attn.to_out.0.")] = value
            elif "adaLN_modulation.1." in k:
                new[k.replace("adaLN_modulation.1.", "norm_out.linear.")] = value
            elif "mlp.fc1." in k:
                new[k.replace("mlp.fc1.", "ff.net.0.proj.")] = value
            elif "mlp.fc2." in k:
                new[k.replace("mlp.fc2.", "ff.net.2.")] = value
            else:
                new[k] = value

        # ------------------------------------------------------------------ #
        # transformer_blocks  (double_blocks → transformer_blocks)
        # ------------------------------------------------------------------ #
        elif key.startswith("double_blocks."):
            k = key.replace("double_blocks.", "transformer_blocks.")

            if "img_attn_qkv" in k:
                to_q, to_k, to_v = value.chunk(3, dim=0)
                suffix = k.split("img_attn_qkv")[1]
                base = k.split("img_attn_qkv")[0]
                new[base + "attn.to_q" + suffix] = to_q
                new[base + "attn.to_k" + suffix] = to_k
                new[base + "attn.to_v" + suffix] = to_v

            elif "txt_attn_qkv" in k:
                to_q, to_k, to_v = value.chunk(3, dim=0)
                suffix = k.split("txt_attn_qkv")[1]
                base = k.split("txt_attn_qkv")[0]
                new[base + "attn.add_q_proj" + suffix] = to_q
                new[base + "attn.add_k_proj" + suffix] = to_k
                new[base + "attn.add_v_proj" + suffix] = to_v

            elif "img_attn_proj." in k:
                new[k.replace("img_attn_proj.", "attn.to_out.0.")] = value
            elif "txt_attn_proj." in k:
                new[k.replace("txt_attn_proj.", "attn.to_add_out.")] = value
            elif "img_attn_q_norm." in k:
                new[k.replace("img_attn_q_norm.", "attn.norm_q.")] = value
            elif "img_attn_k_norm." in k:
                new[k.replace("img_attn_k_norm.", "attn.norm_k.")] = value
            elif "txt_attn_q_norm." in k:
                new[k.replace("txt_attn_q_norm.", "attn.norm_added_q.")] = value
            elif "txt_attn_k_norm." in k:
                new[k.replace("txt_attn_k_norm.", "attn.norm_added_k.")] = value
            elif "img_mod.linear." in k:
                new[k.replace("img_mod.linear.", "norm1.linear.")] = value
            elif "txt_mod.linear." in k:
                new[k.replace("txt_mod.linear.", "norm1_context.linear.")] = value
            elif "img_mlp.fc1." in k:
                new[k.replace("img_mlp.fc1.", "ff.net.0.proj.")] = value
            elif "img_mlp.fc2." in k:
                new[k.replace("img_mlp.fc2.", "ff.net.2.")] = value
            elif "txt_mlp.fc1." in k:
                new[k.replace("txt_mlp.fc1.", "ff_context.net.0.proj.")] = value
            elif "txt_mlp.fc2." in k:
                new[k.replace("txt_mlp.fc2.", "ff_context.net.2.")] = value
            else:
                new[k] = value

        # ------------------------------------------------------------------ #
        # final_layer → norm_out.linear + proj_out
        # The adaLN_modulation weight stores [shift, scale]; diffusers expects
        # [scale, shift], so swap the two halves.
        # ------------------------------------------------------------------ #
        elif key == "final_layer.adaLN_modulation.1.weight":
            shift, scale = value.chunk(2, dim=0)
            new["norm_out.linear.weight"] = torch.cat([scale, shift], dim=0)
        elif key == "final_layer.adaLN_modulation.1.bias":
            shift, scale = value.chunk(2, dim=0)
            new["norm_out.linear.bias"] = torch.cat([scale, shift], dim=0)
        elif key.startswith("final_layer.linear."):
            new[key.replace("final_layer.linear.", "proj_out.")] = value

        # Unrecognised keys are silently dropped.

    return new


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
        the from_single_file whitelist and the GGUF files use Tencent's original
        key names rather than diffusers names, so we load manually:
        1. Download the GGUF file.
        2. Load the raw checkpoint (GGUFParameter tensors).
        3. Remap keys from Tencent format to diffusers format.
        4. Create the model on the meta device and apply the GGUF quantizer.
        5. Load the converted checkpoint into the model.
        """
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from accelerate import init_empty_weights
        from diffusers import GGUFQuantizationConfig, HunyuanVideo15Transformer3DModel
        from diffusers.models.model_loading_utils import (
            load_gguf_checkpoint,
            load_model_dict_into_meta,
        )
        from diffusers.quantizers.auto import DiffusersAutoQuantizer
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        parts = gguf_file.rsplit("/", 1)
        if len(parts) == 2:
            local_path = hf_hub_download(
                repo_id=GGUF_REPO, filename=parts[1], subfolder=parts[0]
            )
        else:
            local_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        raw_checkpoint = load_gguf_checkpoint(local_path)
        checkpoint = _convert_gguf_checkpoint_to_diffusers(raw_checkpoint)

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        hf_quantizer = DiffusersAutoQuantizer.from_config(quantization_config)
        hf_quantizer.validate_environment()
        torch_dtype = hf_quantizer.update_torch_dtype(compute_dtype)

        with init_empty_weights():
            self._transformer = HunyuanVideo15Transformer3DModel()

        hf_quantizer.preprocess_model(
            model=self._transformer,
            device_map=None,
            state_dict=checkpoint,
            keep_in_fp32_modules=[],
        )

        load_model_dict_into_meta(
            self._transformer,
            checkpoint,
            dtype=torch_dtype,
            device_map={"": torch.device("cpu")},
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=[],
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
