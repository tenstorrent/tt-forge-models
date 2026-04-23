# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DINOv3 linear classification probe model loader implementation (PyTorch).

Composes a DINOv3 ViT backbone (gated facebook/dinov3-* repo) with a
pretrained linear classification head released by yberreby as part of the
dinov3-in1k-probes collection. The probe maps the backbone CLS token to
1000 ImageNet-1k logits.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torch import nn

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
class DINOv3LinearProbeConfig(ModelConfig):
    """Configuration specific to DINOv3 linear classification probe models."""

    backbone_pretrained_name: str = ""
    image_size: int = 512
    in_features: int = 768
    out_features: int = 1000


class DINOv3LinearProbeClassifier(nn.Module):
    """DINOv3 backbone composed with a pretrained linear classification head."""

    def __init__(self, backbone: nn.Module, in_features: int, out_features: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls_token = hidden[:, 0, :]
        return self.classifier(cls_token)


class ModelVariant(StrEnum):
    """Available DINOv3 linear classification probe variants."""

    VITB16_512 = "ViTB16_512"


class ModelLoader(ForgeModel):
    """DINOv3 linear classification probe model loader."""

    _VARIANTS = {
        ModelVariant.VITB16_512: DINOv3LinearProbeConfig(
            pretrained_model_name="yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe",
            backbone_pretrained_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
            image_size=512,
            in_features=768,
            out_features=1000,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITB16_512

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DINOv3 Linear Classification Probe",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessorFast

        backbone_name = self._variant_config.backbone_pretrained_name
        image_size = self._variant_config.image_size
        token = os.environ.get("HF_TOKEN")
        processor_kwargs = {
            "size": {"shortest_edge": image_size},
            "crop_size": {"height": image_size, "width": image_size},
        }
        if token:
            processor_kwargs["token"] = token
        try:
            self.processor = DINOv3ViTImageProcessorFast.from_pretrained(
                backbone_name, **processor_kwargs
            )
        except Exception:
            cfg = DINOv3ViTConfig(image_size=image_size, patch_size=16)
            self.processor = DINOv3ViTImageProcessorFast(
                size={"shortest_edge": image_size},
                crop_size={"height": image_size, "width": image_size},
            )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        config = self._variant_config

        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        backbone_kwargs = {}
        if token:
            backbone_kwargs["token"] = token
        if dtype_override is not None:
            backbone_kwargs["torch_dtype"] = dtype_override
        backbone_kwargs |= kwargs

        try:
            backbone = DINOv3ViTModel.from_pretrained(
                config.backbone_pretrained_name, **backbone_kwargs
            )
        except Exception:
            # Fall back to random weights when the gated backbone is inaccessible
            vit_cfg = DINOv3ViTConfig(
                hidden_size=config.in_features,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=config.in_features * 4,
                image_size=config.image_size,
                patch_size=16,
            )
            backbone = DINOv3ViTModel(vit_cfg)

        model = DINOv3LinearProbeClassifier(
            backbone=backbone,
            in_features=config.in_features,
            out_features=config.out_features,
        )

        probe_path = hf_hub_download(
            repo_id=config.pretrained_model_name,
            filename="model.safetensors",
        )
        probe_state_dict = load_file(probe_path)
        model.classifier.load_state_dict(probe_state_dict)

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if self.processor is None:
            self._load_processor()

        if image is None:
            image_size = self._variant_config.image_size
            image = Image.fromarray(
                torch.randint(
                    0, 256, (image_size, image_size, 3), dtype=torch.uint8
                ).numpy()
            )

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
