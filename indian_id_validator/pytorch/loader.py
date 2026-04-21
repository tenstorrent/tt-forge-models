# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Indian ID Validator (logasanjeev/indian-id-validator) model loader implementation.

YOLO11l-based pipeline for Indian identification documents. Includes a
YOLO11l-cls classifier that identifies the document type and a set of
YOLO11l object detectors that locate specific fields on each document
(Aadhaar, PAN Card, Passport, Voter ID, Driving License).
"""
from typing import Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from torchvision import transforms
from datasets import load_dataset

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
    """Available Indian ID Validator model variants."""

    ID_CLASSIFIER = "id_classifier"
    AADHAAR_CARD = "aadhaar_card"
    PAN_CARD = "pan_card"
    PASSPORT = "passport"
    VOTER_ID = "voter_id"
    DRIVING_LICENSE = "driving_license"


class ModelLoader(ForgeModel):
    """Indian ID Validator model loader implementation."""

    _VARIANTS = {
        ModelVariant.ID_CLASSIFIER: ModelConfig(
            pretrained_model_name="models/Id_Classifier.pt",
        ),
        ModelVariant.AADHAAR_CARD: ModelConfig(
            pretrained_model_name="models/Aadhaar_Card.pt",
        ),
        ModelVariant.PAN_CARD: ModelConfig(
            pretrained_model_name="models/Pan_Card.pt",
        ),
        ModelVariant.PASSPORT: ModelConfig(
            pretrained_model_name="models/Passport.pt",
        ),
        ModelVariant.VOTER_ID: ModelConfig(
            pretrained_model_name="models/Voter_Id.pt",
        ),
        ModelVariant.DRIVING_LICENSE: ModelConfig(
            pretrained_model_name="models/Driving_License.pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ID_CLASSIFIER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.CV_IMAGE_CLS
            if variant == ModelVariant.ID_CLASSIFIER
            else ModelTask.CV_OBJECT_DET
        )
        return ModelInfo(
            model="Indian ID Validator",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        filename = self._variant_config.pretrained_model_name
        model_path = hf_hub_download("logasanjeev/indian-id-validator", filename)
        yolo_model = YOLO(model_path)
        model = yolo_model.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        batch_tensor = preprocess(image).unsqueeze(0)
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
