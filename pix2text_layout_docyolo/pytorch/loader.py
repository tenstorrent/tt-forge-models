# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pix2Text Layout DocYOLO (breezedeus/pix2text-layout-docyolo) model loader.

DocLayout-YOLO based document layout detection model forked from
opendatalab/DocLayout-YOLO and used by the Pix2Text (P2T) pipeline.
"""
from typing import Optional

from huggingface_hub import hf_hub_download

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
    """Available Pix2Text Layout DocYOLO model variants."""

    DOCSTRUCTBENCH_IMGSZ1024 = "doclayout_yolo_docstructbench_imgsz1024"


class ModelLoader(ForgeModel):
    """Pix2Text Layout DocYOLO model loader implementation."""

    _VARIANTS = {
        ModelVariant.DOCSTRUCTBENCH_IMGSZ1024: ModelConfig(
            pretrained_model_name="doclayout_yolo_docstructbench_imgsz1024.pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOCSTRUCTBENCH_IMGSZ1024

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pix2Text Layout DocYOLO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from doclayout_yolo import YOLOv10

        filename = self._variant_config.pretrained_model_name
        model_path = hf_hub_download("breezedeus/pix2text-layout-docyolo", filename)
        yolo_model = YOLOv10(model_path)
        model = yolo_model.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from datasets import load_dataset
        from torchvision import transforms

        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"].convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ]
        )
        batch_tensor = preprocess(image).unsqueeze(0)
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
