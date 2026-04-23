# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXS-512-tinySDdistilled GGUF model loader implementation.

Loads the GGUF-quantized UNet from concedo/sdxs-512-tinySDdistilled-GGUF and
builds a text-to-image pipeline using IDKiro/sdxs-512-dreamshaper as the base
model for the remaining components.

SDXS is a one-step latent diffusion model distilled from Stable Diffusion 1.5
for fast 512x512 text-to-image generation.

Available variants:
- Q8_0: 8-bit quantization (~683 MB)
"""

from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

GGUF_REPO = "concedo/sdxs-512-tinySDdistilled-GGUF"
BASE_PIPELINE = "IDKiro/sdxs-512-dreamshaper"


class ModelVariant(StrEnum):
    """Available SDXS-512-tinySDdistilled GGUF variants."""

    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q8_0: "sdxs-512-tinySDdistilled_Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """SDXS-512-tinySDdistilled GGUF model loader."""

    _VARIANTS = {
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXS_512_tinySDdistilled_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _convert_gguf_unet_checkpoint(raw_checkpoint, layers_per_block=1):
        """Convert raw GGUF checkpoint to diffusers UNet state dict.

        Handles the non-contiguous block indices in the tinySDdistilled GGUF
        (exported with SD1.5-style numbering where every other resnet slot is
        empty due to layers_per_block=1 instead of 2).

        Also handles the mixed-format GGUF where some blocks use LDM-format
        keys and up_blocks.0 attentions use diffusers-format keys directly.
        """
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

        ldm_prefix = "model.diffusion_model."

        # Strip LDM prefix and dequantize all GGUFParameter tensors to float.
        # The GGUF stores GroupNorm weights as Q8_0 (packed shape differs from
        # model shape), so we dequantize everything for compatibility.
        unet_sd = {}
        for key, val in raw_checkpoint.items():
            if not key.startswith(ldm_prefix):
                continue
            short = key[len(ldm_prefix) :]
            if hasattr(val, "quant_type"):
                val = dequantize_gguf_tensor(val)
            unet_sd[short] = val

        new_sd = {}

        # Static key remappings
        static = {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "conv_norm_out.weight": "out.0.weight",
            "conv_norm_out.bias": "out.0.bias",
            "conv_out.weight": "out.2.weight",
            "conv_out.bias": "out.2.bias",
        }
        for diff_key, ldm_key in static.items():
            if ldm_key in unet_sd:
                new_sd[diff_key] = unet_sd[ldm_key]

        # Down blocks — sparse input_block indices due to tinySD layers_per_block=1.
        # The GGUF uses SD1.5 numbering (skipping every 2nd resnet slot), so the
        # actual indices {1,3,4,6,7} map via a virtual contiguous sequence.
        input_indices = sorted(
            set(int(k.split(".")[1]) for k in unet_sd if k.startswith("input_blocks."))
        )
        non_zero = [i for i in input_indices if i > 0]

        for virtual_i, actual_i in enumerate(non_zero, start=1):
            block_id = (virtual_i - 1) // (layers_per_block + 1)
            layer_id = (virtual_i - 1) % (layers_per_block + 1)

            pfx = f"input_blocks.{actual_i}"
            new_pfx_r = f"down_blocks.{block_id}.resnets.{layer_id}"
            new_pfx_a = f"down_blocks.{block_id}.attentions.{layer_id}"

            for k in [
                k for k in unet_sd if k.startswith(f"{pfx}.0.") and ".op." not in k
            ]:
                new_k = (
                    k.replace("in_layers.0", "norm1")
                    .replace("in_layers.2", "conv1")
                    .replace("out_layers.0", "norm2")
                    .replace("out_layers.3", "conv2")
                    .replace("emb_layers.1", "time_emb_proj")
                    .replace("skip_connection", "conv_shortcut")
                    .replace(f"{pfx}.0", new_pfx_r)
                )
                new_sd[new_k] = unet_sd[k]

            if f"{pfx}.0.op.weight" in unet_sd:
                new_sd[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_sd[
                    f"{pfx}.0.op.weight"
                ]
                new_sd[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_sd[
                    f"{pfx}.0.op.bias"
                ]

            for k in [k for k in unet_sd if k.startswith(f"{pfx}.1.")]:
                new_sd[k.replace(f"{pfx}.1", new_pfx_a)] = unet_sd[k]

        # Up blocks — output_block indices 0-7 are contiguous but blocks 2 and 5
        # are standalone upsamplers (no resnet), which shifts block_id calculation.
        output_indices = sorted(
            set(int(k.split(".")[1]) for k in unet_sd if k.startswith("output_blocks."))
        )

        resnet_count = 0
        for actual_i in output_indices:
            pfx = f"output_blocks.{actual_i}"
            resnet_keys = [
                k for k in unet_sd if k.startswith(f"{pfx}.0.") and ".op." not in k
            ]
            has_resnet = bool(resnet_keys)

            if has_resnet:
                block_id = resnet_count // (layers_per_block + 1)
                layer_id = resnet_count % (layers_per_block + 1)
                resnet_count += 1
            else:
                block_id = (resnet_count - 1) // (layers_per_block + 1)
                layer_id = 0  # unused for upsampler-only blocks

            new_pfx_r = f"up_blocks.{block_id}.resnets.{layer_id}"
            new_pfx_a = f"up_blocks.{block_id}.attentions.{layer_id}"

            for k in resnet_keys:
                new_k = (
                    k.replace("in_layers.0", "norm1")
                    .replace("in_layers.2", "conv1")
                    .replace("out_layers.0", "norm2")
                    .replace("out_layers.3", "conv2")
                    .replace("emb_layers.1", "time_emb_proj")
                    .replace("skip_connection", "conv_shortcut")
                    .replace(f"{pfx}.0", new_pfx_r)
                )
                new_sd[new_k] = unet_sd[k]

            for k in [
                k for k in unet_sd if k.startswith(f"{pfx}.1.") and ".conv." not in k
            ]:
                new_sd[k.replace(f"{pfx}.1", new_pfx_a)] = unet_sd[k]

            if f"{pfx}.1.conv.weight" in unet_sd:
                new_sd[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_sd[
                    f"{pfx}.1.conv.weight"
                ]
                new_sd[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_sd[
                    f"{pfx}.1.conv.bias"
                ]
            if f"{pfx}.2.conv.weight" in unet_sd:
                new_sd[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_sd[
                    f"{pfx}.2.conv.weight"
                ]
                new_sd[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_sd[
                    f"{pfx}.2.conv.bias"
                ]

        # Pass through keys already in diffusers format (e.g. up_blocks.0.attentions.*)
        for k in unet_sd:
            if k.startswith(("up_blocks.", "down_blocks.", "mid_block.")):
                if k not in new_sd:
                    new_sd[k] = unet_sd[k]

        return new_sd

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized UNet and build the SDXS pipeline.

        Manually converts the GGUF checkpoint to diffusers format to work
        around a bug in diffusers 0.37.1 where convert_ldm_unet_checkpoint
        fails on non-contiguous block indices produced by tinySDdistilled's
        layers_per_block=1 architecture exported with SD1.5-style numbering.

        Returns the UNet module (not the full pipeline) for compatibility with
        the test infrastructure.
        """
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel
        from diffusers.models.model_loading_utils import load_gguf_checkpoint
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        raw_checkpoint = load_gguf_checkpoint(gguf_path)
        converted_sd = self._convert_gguf_unet_checkpoint(
            raw_checkpoint, layers_per_block=1
        )

        unet = UNet2DConditionModel.from_pretrained(
            BASE_PIPELINE,
            subfolder="unet",
            torch_dtype=compute_dtype,
        )
        unet.load_state_dict(converted_sd, strict=False)
        unet = unet.to(compute_dtype)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            unet=unet,
            torch_dtype=compute_dtype,
        )

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the SDXS UNet.

        Returns a dict of keyword arguments for UNet.forward():
            sample: random noise latent [batch, 4, 64, 64]
            timestep: single timestep tensor
            encoder_hidden_states: CLIP text embeddings [batch, 77, 768]
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        prompt = [
            "a close-up picture of an old man standing in the rain",
        ] * batch_size

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(
                text_inputs.input_ids
            ).last_hidden_state.to(compute_dtype)

        latent = torch.randn(batch_size, 4, 64, 64, dtype=compute_dtype)
        timestep = torch.tensor([999], dtype=torch.long)

        return {
            "sample": latent,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
