# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CheXpert-MIMIC CXR Impression Baseline model loader implementation (PyTorch).
"""

import torch
from transformers import (
    BertTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)
from typing import Optional
from PIL import Image

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available CheXpert-MIMIC CXR Impression Baseline model variants."""

    BASELINE = "Baseline"


class ModelLoader(ForgeModel):
    """CheXpert-MIMIC CXR Impression Baseline model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASELINE: ModelConfig(
            pretrained_model_name="IAMJB/chexpert-mimic-cxr-impression-baseline",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASELINE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CheXpert-MIMIC-CXR-Impression-Baseline",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name)

        image_path = get_file(
            "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
        )
        image = Image.open(str(image_path)).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

        return {"pixel_values": pixel_values}

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.tokenizer is None:
            pretrained_model_name = self._variant_config.pretrained_model_name
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
