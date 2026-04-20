# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DocTR PARSeq Multilingual PyTorch model loader for text recognition.

This model is a PARSeq (Permuted Autoregressive Sequence) text recognition
model from the docTR library, supporting 12 Latin-script languages
(English, Danish, French, Italian, Spanish, German, Portuguese, Czech,
Polish, Dutch, Norwegian, Finnish).
"""
from typing import Optional

from PIL import Image
from torchvision import transforms

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DocTR PARSeq Multilingual model variants."""

    PARSEQ_MULTILINGUAL_V1 = "parseq-multilingual-v1"


class ModelLoader(ForgeModel):
    """DocTR PARSeq Multilingual PyTorch model loader for text recognition."""

    _VARIANTS = {
        ModelVariant.PARSEQ_MULTILINGUAL_V1: ModelConfig(
            pretrained_model_name="Felix92/doctr-torch-parseq-multilingual-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PARSEQ_MULTILINGUAL_V1

    # Model input preprocessing parameters from config.json
    INPUT_HEIGHT = 32
    INPUT_WIDTH = 128
    MEAN = [0.694, 0.695, 0.693]
    STD = [0.299, 0.296, 0.301]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="doctr_parseq_multilingual",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the DocTR PARSeq multilingual recognition model from HuggingFace Hub."""
        from doctr.models import from_hub

        model = from_hub(self._variant_config.pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, *, dtype_override=None, batch_size=1):
        """Prepare sample input for the PARSeq text recognition model.

        Returns a synthetic text-like image tensor of shape
        [batch, 3, 32, 128], normalized with the model's expected mean/std.
        """
        image = Image.new(
            "RGB", (self.INPUT_WIDTH, self.INPUT_HEIGHT), color=(200, 200, 200)
        )

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD),
            ]
        )
        inputs = preprocess(image).unsqueeze(0)
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
