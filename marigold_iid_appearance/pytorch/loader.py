# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Marigold IID Appearance model loader implementation for intrinsic image decomposition.
"""
import torch
from diffusers import MarigoldIntrinsicsPipeline
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Marigold IID Appearance model variants."""

    V1_1 = "v1.1"


class ModelLoader(ForgeModel):
    """Marigold IID Appearance model loader implementation."""

    _VARIANTS = {
        ModelVariant.V1_1: ModelConfig(
            pretrained_model_name="prs-eth/marigold-iid-appearance-v1-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MarigoldIIDAppearance",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name
        pipe = MarigoldIntrinsicsPipeline.from_pretrained(
            pretrained_model_name, torch_dtype=dtype_override or torch.float32
        )
        pipe.to("cpu")

        for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        self.pipeline = pipe
        return pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32

        prompt = ""
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[0].to(
            dtype=dtype
        )

        latent_size = self.pipeline.unet.config.sample_size
        num_channels = self.pipeline.unet.config.in_channels

        sample = torch.randn(
            batch_size,
            num_channels,
            latent_size,
            latent_size,
            dtype=dtype,
        )

        timestep = torch.tensor([1], dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
