# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXS-512-tinySDdistilled GGUF model loader implementation.

Loads the GGUF checkpoint from concedo/sdxs-512-tinySDdistilled-GGUF and
applies dequantized UNet weights onto the IDKiro/sdxs-512-dreamshaper pipeline,
which provides the correct tinySD architecture (3-stage pruned SD1.5).

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
BASE_REPO = "IDKiro/sdxs-512-dreamshaper"


class ModelVariant(StrEnum):
    """Available SDXS-512-tinySDdistilled GGUF variants."""

    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q8_0: "sdxs-512-tinySDdistilled_Q8_0.gguf",
}

# Maps sparse tinySD GGUF block prefixes to (diffusers prefix, block kind).
# The GGUF uses original SD1.5 block numbering with pruned blocks removed but
# indices kept, so block indices are non-contiguous ([0,1,3,4,6,7] for down).
_BLOCK_PREFIX_MAP = {
    # down path (DownBlock2D, CrossAttnDownBlock2D x2)
    "input_blocks.1.0": ("down_blocks.0.resnets.0", "resnet"),
    "input_blocks.3.0": ("down_blocks.0.downsamplers.0", "downsampler"),
    "input_blocks.4.0": ("down_blocks.1.resnets.0", "resnet"),
    "input_blocks.4.1": ("down_blocks.1.attentions.0", "attn"),
    "input_blocks.6.0": ("down_blocks.1.downsamplers.0", "downsampler"),
    "input_blocks.7.0": ("down_blocks.2.resnets.0", "resnet"),
    "input_blocks.7.1": ("down_blocks.2.attentions.0", "attn"),
    # mid block
    "middle_block.0": ("mid_block.resnets.0", "resnet"),
    "middle_block.1": ("mid_block.attentions.0", "attn"),
    "middle_block.2": ("mid_block.resnets.1", "resnet"),
    # up path — GGUF retains original SD1.5 indices (3 resnets/stage from
    # layers_per_block=2); IDKiro uses layers_per_block=1 (2 resnets/stage),
    # so resnets.2 entries won't exist and will be silently skipped.
    "output_blocks.0.0": ("up_blocks.0.resnets.0", "resnet"),
    "output_blocks.1.0": ("up_blocks.0.resnets.1", "resnet"),
    "output_blocks.2.0": ("up_blocks.0.resnets.2", "resnet"),
    "output_blocks.2.1": ("up_blocks.0.upsamplers.0", "upsampler"),
    "output_blocks.3.0": ("up_blocks.1.resnets.0", "resnet"),
    "output_blocks.3.1": ("up_blocks.1.attentions.0", "attn"),
    "output_blocks.4.0": ("up_blocks.1.resnets.1", "resnet"),
    "output_blocks.4.1": ("up_blocks.1.attentions.1", "attn"),
    "output_blocks.5.0": ("up_blocks.1.resnets.2", "resnet"),
    "output_blocks.5.1": ("up_blocks.1.attentions.2", "attn"),
    "output_blocks.5.2": ("up_blocks.1.upsamplers.0", "upsampler"),
    "output_blocks.6.0": ("up_blocks.2.resnets.0", "resnet"),
    "output_blocks.7.0": ("up_blocks.2.resnets.1", "resnet"),
}

# Sorted by descending prefix length so more-specific prefixes match first.
_SORTED_PREFIX_MAP = sorted(_BLOCK_PREFIX_MAP.items(), key=lambda x: -len(x[0]))

# LDM resnet sub-key → diffusers naming.
_RESNET_SUBKEY_MAP = [
    ("in_layers.0.", "norm1."),
    ("in_layers.2.", "conv1."),
    ("out_layers.0.", "norm2."),
    ("out_layers.3.", "conv2."),
    ("emb_layers.1.", "time_emb_proj."),
    ("skip_connection.", "conv_shortcut."),
]


def _remap_resnet_rest(rest):
    for old, new in _RESNET_SUBKEY_MAP:
        if rest.startswith(old):
            return new + rest[len(old) :]
    return rest


def _gguf_to_diffusers_key(local_key):
    """Map model.diffusion_model.* suffix to a diffusers UNet2DConditionModel key."""
    if local_key.startswith("time_embed.0."):
        return "time_embedding.linear_1." + local_key[len("time_embed.0.") :]
    if local_key.startswith("time_embed.2."):
        return "time_embedding.linear_2." + local_key[len("time_embed.2.") :]
    if local_key.startswith("out.0."):
        return "conv_norm_out." + local_key[len("out.0.") :]
    if local_key.startswith("out.2."):
        return "conv_out." + local_key[len("out.2.") :]
    if local_key.startswith("input_blocks.0.0."):
        return "conv_in." + local_key[len("input_blocks.0.0.") :]

    for prefix, (diffusers_prefix, kind) in _SORTED_PREFIX_MAP:
        dot_prefix = prefix + "."
        if local_key.startswith(dot_prefix):
            rest = local_key[len(dot_prefix) :]
            if kind == "resnet":
                rest = _remap_resnet_rest(rest)
            elif kind == "downsampler":
                # LDM uses `op.` for the conv; diffusers uses `conv.`
                if rest.startswith("op."):
                    rest = "conv." + rest[len("op.") :]
            # attn and upsampler sub-keys map 1:1
            return diffusers_prefix + "." + rest

    return None


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

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the SDXS tinySDdistilled GGUF checkpoint as a full pipeline.

        Loads the IDKiro/sdxs-512-dreamshaper base pipeline (correct tinySD
        architecture: 3-stage pruned SD1.5 with layers_per_block=1) then
        overwrites the UNet weights with dequantized tensors from the GGUF
        file using a custom sparse-index-aware key mapping.  The GGUF retains
        original SD1.5 block numbering so block indices are non-contiguous.
        """
        import gguf
        import numpy as np
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_REPO,
            torch_dtype=compute_dtype,
        )

        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
        reader = gguf.GGUFReader(gguf_path)

        prefix = "model.diffusion_model."
        unet_state = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith(prefix):
                continue
            local_key = tensor.name[len(prefix) :]
            diffusers_key = _gguf_to_diffusers_key(local_key)
            if diffusers_key is None:
                continue
            arr = gguf.dequantize(tensor.data, tensor.tensor_type)
            unet_state[diffusers_key] = torch.from_numpy(np.array(arr)).to(
                compute_dtype
            )

        self.pipeline.unet.load_state_dict(unet_state, strict=False)

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for SDXS.

        Returns:
            list: A list of sample text prompts.
        """
        return ["a close-up picture of an old man standing in the rain"] * batch_size
