# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LFM2-VL ONNX model loader implementation for multimodal image-text-to-text
generation.

Loads onnx-community/LFM2-VL-450M-ONNX vision encoder component, a lightweight
450M-parameter multimodal vision-language model by Liquid AI combining a
SigLIP2 NaFlex base vision tower with the LFM2-350M language backbone.
"""

from typing import Optional

import onnx
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor

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
from ....tools.utils import get_file

_REPO_ID = "onnx-community/LFM2-VL-450M-ONNX"


class ModelVariant(StrEnum):
    """Available LFM2-VL ONNX model variants."""

    LFM2_VL_450M_ONNX = "LFM2_VL_450M_ONNX"


class ModelLoader(ForgeModel):
    """LFM2-VL ONNX model loader for multimodal image-text-to-text generation tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_VL_450M_ONNX: ModelConfig(
            pretrained_model_name=_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_VL_450M_ONNX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LFM2-VL ONNX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, **kwargs):
        """Load and return the LFM2-VL ONNX vision encoder model.

        Returns:
            onnx.ModelProto: The loaded ONNX vision encoder model.
        """
        onnx_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="onnx/vision_encoder.onnx",
        )
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the LFM2-VL ONNX vision encoder.

        Returns:
            dict: Input tensors for the vision encoder model.
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

        return {"pixel_values": inputs["pixel_values"]}
