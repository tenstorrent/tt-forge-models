# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pancancer Lymphocytes InceptionV4 model loader for tumor infiltrating lymphocyte classification.
"""

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
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


class ModelVariant(StrEnum):
    """Available Pancancer Lymphocytes InceptionV4 model variants."""

    TCGA = "TCGA"


class ModelLoader(ForgeModel):
    """Pancancer Lymphocytes InceptionV4 model loader for TIL histopathology classification."""

    _VARIANTS = {
        ModelVariant.TCGA: ModelConfig(
            pretrained_model_name="kaczmarj/pancancer-lymphocytes-inceptionv4.tcga",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TCGA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Pancancer Lymphocytes InceptionV4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Pancancer Lymphocytes InceptionV4 TorchScript model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The InceptionV4 (no batch norm) model instance.
        """
        jit_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="torchscript_model.pt",
        )
        model = torch.jit.load(jit_path)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the input tensor.

        Returns:
            torch.Tensor: Preprocessed input tensor of shape [batch, 3, 299, 299].
        """
        preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        image = Image.new("RGB", (299, 299), (200, 150, 180))
        input_tensor = preprocess(image).unsqueeze(0)

        if batch_size > 1:
            input_tensor = input_tensor.expand(batch_size, -1, -1, -1).contiguous()

        if dtype_override is not None:
            input_tensor = input_tensor.to(dtype_override)

        return input_tensor
