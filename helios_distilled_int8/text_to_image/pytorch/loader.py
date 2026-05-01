# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helios-Distilled-int8 model loader implementation for text-to-image generation.

The szwagros/Helios-Distilled-int8 HuggingFace repo stores transformer and
text-encoder weights as flat top-level safetensors files rather than in the
standard diffusers sub-directory layout (transformer/config.json etc.).
DiffusionPipeline.from_pretrained() fails with OSError because it cannot find
config.json in the expected sub-directories.

Fix: load sub-model configs from the reference BestWishYsh/Helios-Distilled
repo and weights from the flat safetensors files in the int8 repo, then
assemble the HeliosPyramidPipeline manually.
"""

import json
from typing import Optional, Dict, Any

import torch
from diffusers import DiffusionPipeline

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# Reference repo that has the canonical sub-directory layout with config.json
_REFERENCE_REPO = "BestWishYsh/Helios-Distilled"
# Int8 repo that has flat safetensors but no sub-directory configs
_INT8_REPO = "szwagros/Helios-Distilled-int8"


class ModelVariant(StrEnum):
    """Available Helios-Distilled-int8 model variants."""

    HELIOS_DISTILLED_INT8 = "Helios-Distilled-int8"


class ModelLoader(ForgeModel):
    """Helios-Distilled-int8 model loader implementation."""

    _VARIANTS = {
        ModelVariant.HELIOS_DISTILLED_INT8: ModelConfig(
            pretrained_model_name=_INT8_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HELIOS_DISTILLED_INT8

    DEFAULT_PROMPT = "A cinematic portrait of a robot in a neon-lit lab"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Helios-Distilled",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
    ) -> DiffusionPipeline:
        """Manually construct HeliosPyramidPipeline from the int8 repo.

        The int8 repo (szwagros/Helios-Distilled-int8) stores model weights as
        flat top-level safetensors files.  DiffusionPipeline.from_pretrained()
        cannot load it because it looks for transformer/config.json and
        text_encoder/config.json sub-directories which are absent.

        Strategy:
          - Transformer: load config from _REFERENCE_REPO, weights from
            Helios-Distilled-int8.safetensors in the int8 repo.
          - Text encoder: load config from _REFERENCE_REPO, weights from
            Helios-umt5-xxl-int8.safetensors in the int8 repo.
          - VAE, tokenizer, scheduler: load directly from the int8 repo
            (these sub-directories exist and have the standard layout).
        """
        from diffusers import AutoencoderKLWan, HeliosPyramidPipeline, HeliosTransformer3DModel
        from diffusers.schedulers import HeliosDMDScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import UMT5EncoderModel, T5TokenizerFast

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # ------------------------------------------------------------------
        # 1. Transformer
        # ------------------------------------------------------------------
        transformer_config_path = hf_hub_download(
            _REFERENCE_REPO, "transformer/config.json"
        )
        with open(transformer_config_path) as f:
            transformer_config = json.load(f)

        transformer = HeliosTransformer3DModel.from_config(transformer_config)
        transformer_weights_path = hf_hub_download(
            _INT8_REPO, "Helios-Distilled-int8.safetensors"
        )
        transformer_state_dict = load_file(transformer_weights_path)
        transformer.load_state_dict(transformer_state_dict)
        transformer = transformer.to(dtype=dtype)

        # ------------------------------------------------------------------
        # 2. Text encoder (UMT5-XXL int8)
        # ------------------------------------------------------------------
        text_encoder_config_path = hf_hub_download(
            _REFERENCE_REPO, "text_encoder/config.json"
        )
        with open(text_encoder_config_path) as f:
            text_encoder_config_dict = json.load(f)
        from transformers import UMT5Config
        text_encoder_config = UMT5Config(**{
            k: v for k, v in text_encoder_config_dict.items()
            if not k.startswith("_")
        })
        text_encoder = UMT5EncoderModel(text_encoder_config)
        text_encoder_weights_path = hf_hub_download(
            _INT8_REPO, "Helios-umt5-xxl-int8.safetensors"
        )
        text_encoder_state_dict = load_file(text_encoder_weights_path)
        text_encoder.load_state_dict(text_encoder_state_dict)
        text_encoder = text_encoder.to(dtype=dtype)

        # ------------------------------------------------------------------
        # 3. VAE, tokenizer, scheduler from the int8 repo (standard layout)
        # ------------------------------------------------------------------
        vae = AutoencoderKLWan.from_pretrained(
            _INT8_REPO, subfolder="vae", torch_dtype=dtype
        )
        tokenizer = T5TokenizerFast.from_pretrained(
            _INT8_REPO, subfolder="tokenizer"
        )
        scheduler = HeliosDMDScheduler.from_pretrained(
            _INT8_REPO, subfolder="scheduler"
        )

        # ------------------------------------------------------------------
        # 4. Assemble the pipeline
        # ------------------------------------------------------------------
        self.pipeline = HeliosPyramidPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            is_distilled=True,
        )

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """
        Load and return the Helios-Distilled-int8 text-to-image pipeline.
        """
        if self.pipeline is None:
            return self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
