# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin feature extraction model loader.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

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


@dataclass
class SwinFEConfig(ModelConfig):
    """Configuration specific to Swin feature extraction models."""

    source: ModelSource = ModelSource.HUGGING_FACE


class ModelVariant(StrEnum):
    """Available Swin feature extraction model variants."""

    TINY_RANDOM = "TinyRandom"
    SWIN_TINY_PATCH4_WINDOW7_224_CTRANSPATH = "Tiny_Patch4_Window7_224_CTransPath"


class ConvStem(nn.Module):
    """Custom patch embedding used by CTransPath Swin models.

    Adapted from https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        **kwargs,
    ):
        super().__init__()
        from timm.layers.helpers import to_2tuple

        assert patch_size == 4, "Patch size must be 4"
        assert embed_dim % 8 == 0, "Embedding dimension must be a multiple of 8"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class ModelLoader(ForgeModel):
    """Swin feature extraction model loader."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: SwinFEConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-SwinModel",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.SWIN_TINY_PATCH4_WINDOW7_224_CTRANSPATH: SwinFEConfig(
            pretrained_model_name="hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
            source=ModelSource.TIMM,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="Swin",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=source,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            import timm

            model = timm.create_model(
                pretrained_model_name,
                embed_layer=ConvStem,
                pretrained=True,
            )
        else:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        model.eval()

        if dtype_override is not None and source == ModelSource.TIMM:
            model = model.to(dtype_override)

        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        source = self._variant_config.source

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        if source == ModelSource.TIMM:
            from timm.data import create_transform, resolve_data_config

            model_for_config = self._cached_model
            if model_for_config is None:
                model_for_config = self.load_model(dtype_override=dtype_override)

            data_config = resolve_data_config({}, model=model_for_config)
            transforms = create_transform(**data_config, is_training=False)
            inputs = transforms(image).unsqueeze(0)
            inputs = inputs.repeat_interleave(batch_size, dim=0)

            if dtype_override is not None:
                inputs = inputs.to(dtype_override)
            return inputs

        if self.processor is None:
            self._load_processor()

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
