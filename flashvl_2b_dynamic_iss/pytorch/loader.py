# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FlashVL-2B-Dynamic-ISS model loader implementation for multimodal visual question answering.
"""

from typing import Optional

from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available FlashVL-2B-Dynamic-ISS model variants."""

    FLASHVL_2B_DYNAMIC_ISS = "2B_Dynamic_ISS"


class ModelLoader(ForgeModel):
    """FlashVL-2B-Dynamic-ISS model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.FLASHVL_2B_DYNAMIC_ISS: ModelConfig(
            pretrained_model_name="FlashVL/FlashVL-2B-Dynamic-ISS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLASHVL_2B_DYNAMIC_ISS

    default_query = "Describe this image."
    default_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FlashVL-2B-Dynamic-ISS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processors(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FlashVL-2B-Dynamic-ISS model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None or self.image_processor is None:
            self._load_processors()

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.tokenizer = self.tokenizer
        model.im_trans = self.image_processor
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for FlashVL-2B-Dynamic-ISS."""
        if self.tokenizer is None or self.image_processor is None:
            self._load_processors()

        image_file = get_file(self.default_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [{"role": "user", "content": self.default_query}]

        return {"pil_image": image, "messages": messages}
