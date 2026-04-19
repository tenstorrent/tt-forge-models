# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiomedParse model loader implementation for biomedical image segmentation.

BiomedParse is a biomedical foundation model that jointly performs image
segmentation, object detection, and object recognition across 9 biomedical
imaging modalities (CT, MRI, X-Ray, Pathology, etc.) using text-guided prompts.

Reference: https://github.com/microsoft/BiomedParse
HuggingFace: https://huggingface.co/microsoft/BiomedParse
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms

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
from datasets import load_dataset


class BiomedParseWrapper(nn.Module):
    """Wraps BiomedParse to accept a plain image tensor instead of a list of dicts."""

    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model
        self.backbone = inner_model.backbone
        self.sem_seg_head = inner_model.sem_seg_head
        self.register_buffer("pixel_mean", inner_model.pixel_mean.clone())
        self.register_buffer("pixel_std", inner_model.pixel_std.clone())
        self.size_divisibility = inner_model.size_divisibility

    def _ensure_text_embeddings(self):
        """Pre-set fake text embeddings so the language encoder doesn't need CUDA."""
        lang_enc = self.sem_seg_head.predictor.lang_encoder
        if not hasattr(lang_enc, "default_text_embeddings"):
            num_classes = 17
            embed_dim = 512
            fake_emb = torch.randn(num_classes, embed_dim, dtype=self.pixel_mean.dtype)
            fake_emb = fake_emb / fake_emb.norm(dim=-1, keepdim=True)
            lang_enc.default_text_embeddings = fake_emb

    def forward(self, images):
        from detectron2.structures import ImageList

        self._ensure_text_embeddings()
        images = (images - self.pixel_mean) / self.pixel_std
        images = ImageList.from_tensors(
            [images[i] for i in range(images.shape[0])],
            self.size_divisibility,
        )
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        return outputs["pred_masks"]


class ModelVariant(StrEnum):
    """Available BiomedParse model variants."""

    V1 = "v1"


class ModelLoader(ForgeModel):
    """BiomedParse model loader for biomedical image segmentation."""

    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name="microsoft/BiomedParse",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BiomedParse",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BiomedParse model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BiomedParse model instance.
        """
        from .src.model import build_biomedparse_model

        base_model = build_biomedparse_model()
        model = BiomedParseWrapper(base_model.model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BiomedParse model.

        BiomedParse expects a 1024x1024 RGB image tensor normalized with
        ImageNet statistics.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (batch_size, 3, 1024, 1024).
        """
        image_size = (512, 512)
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Load sample image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = transform(image).unsqueeze(0)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
