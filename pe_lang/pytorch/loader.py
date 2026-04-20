# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PE-Lang (Perception Encoder - Language aligned) model loader implementation
for image feature extraction.
"""
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available PE-Lang model variants."""

    G14_448 = "G14_448"


class ModelLoader(ForgeModel):
    """PE-Lang model loader using Meta's perception-encoder for image feature extraction."""

    _VARIANTS = {
        ModelVariant.G14_448: ModelConfig(
            pretrained_model_name="PE-Lang-G14-448",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.G14_448

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PE-Lang",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PE-Lang model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The PE-Lang VisionTransformer model instance.
        """
        from core.vision_encoder import pe, transforms

        model_name = self._variant_config.pretrained_model_name

        model = pe.VisionTransformer.from_config(model_name, pretrained=True)
        self.preprocess = transforms.get_image_transform(model.image_size)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the PE-Lang model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensor containing the preprocessed image.
        """
        from core.vision_encoder import pe, transforms

        if self.preprocess is None:
            model_name = self._variant_config.pretrained_model_name
            model = pe.VisionTransformer.from_config(model_name, pretrained=False)
            self.preprocess = transforms.get_image_transform(model.image_size)

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        pixel_values = self.preprocess(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"image": pixel_values}
