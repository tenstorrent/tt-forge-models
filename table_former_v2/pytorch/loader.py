# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TableFormerV2 model loader implementation for table structure recognition.
"""
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
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


IMAGE_SIZE = 448


class ModelVariant(StrEnum):
    """Available TableFormerV2 model variants for table structure recognition."""

    TABLE_FORMER_V2 = "TableFormerV2"


class ModelLoader(ForgeModel):
    """TableFormerV2 model loader for table structure recognition tasks."""

    _VARIANTS = {
        ModelVariant.TABLE_FORMER_V2: ModelConfig(
            pretrained_model_name="docling-project/TableFormerV2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TABLE_FORMER_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TableFormerV2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # Importing registers TableFormerV2Config/TableFormerV2 with AutoConfig/AutoModel.
        import docling_ibm_models.tableformer_v2  # noqa: F401

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        preprocess = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(255, 255, 255))
        images = preprocess(image).unsqueeze(0)

        # <start> token (id 2) begins the autoregressive structure decoding.
        input_ids = torch.tensor([[2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        if batch_size > 1:
            images = images.repeat_interleave(batch_size, dim=0)
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            images = images.to(dtype_override)

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
