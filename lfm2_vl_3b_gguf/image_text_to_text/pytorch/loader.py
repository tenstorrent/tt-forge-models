# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LFM2-VL 3B GGUF model loader implementation for image-text-to-text tasks.
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from transformers.image_utils import load_image
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available LFM2-VL 3B GGUF model variants for image-text-to-text tasks."""

    LFM2_VL_3B_GGUF = "3B_GGUF"
    LFM2_VL_1_6B_GGUF = "1.6B_GGUF"


class ModelLoader(ForgeModel):
    """LFM2-VL 3B GGUF model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_VL_3B_GGUF: LLMModelConfig(
            pretrained_model_name="ZuzeTt/LFM2-VL-3B-heretic-Imatrix-GGUF",
            max_length=128,
        ),
        ModelVariant.LFM2_VL_1_6B_GGUF: LLMModelConfig(
            pretrained_model_name="LiquidAI/LFM2-VL-1.6B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_VL_3B_GGUF

    _GGUF_FILES = {
        ModelVariant.LFM2_VL_3B_GGUF: "LFM2-VL-3B-heretic-imatrix-Q4_K_M.gguf",
        ModelVariant.LFM2_VL_1_6B_GGUF: "LFM2-VL-1.6B-Q4_0.gguf",
    }

    _PROCESSOR_NAMES = {
        ModelVariant.LFM2_VL_3B_GGUF: "LiquidAI/LFM2-VL-3B",
        ModelVariant.LFM2_VL_1_6B_GGUF: "LiquidAI/LFM2-VL-1.6B",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LFM2-VL 3B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._PROCESSOR_NAMES[self._variant], **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # gguf is installed dynamically by RequirementsManager after transformers is already
        # imported, so transformers' PACKAGE_DISTRIBUTION_MAPPING cache is stale. Refresh it
        # so is_gguf_available() can find the installed gguf metadata.
        import importlib.metadata
        import transformers.utils.import_utils as _tuu

        _tuu.PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # The GGUF metadata identifies the architecture as "lfm2" (base LM), but
        # AutoModelForImageTextToText needs Lfm2VlConfig. Load the VL config explicitly
        # from the non-GGUF model so transformers uses the correct model class.
        vl_config = AutoConfig.from_pretrained(self._PROCESSOR_NAMES[self._variant])

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file
        model_kwargs["config"] = vl_config

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.gguf_file,
        )
        return self.config
