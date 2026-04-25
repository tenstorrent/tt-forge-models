# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Janus model loader implementation for multimodal understanding.
"""

from typing import Optional

import torch
from PIL import Image
from janus.models import MultiModalityCausalLM  # registers multi_modality type
from transformers import AutoModelForCausalLM, AutoProcessor

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
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available DeepSeek Janus model variants."""

    JANUS_1_3B = "Janus_1_3B"


class ModelLoader(ForgeModel):
    """DeepSeek Janus model loader for multimodal understanding."""

    _VARIANTS = {
        ModelVariant.JANUS_1_3B: ModelConfig(
            pretrained_model_name="deepseek-ai/Janus-1.3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JANUS_1_3B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Janus model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Janus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Janus model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x always uses torch.device("meta") in get_init_context,
        # but siglip_vit.py calls item() during __init__ which fails on meta tensors.
        # Patch get_init_context to exclude the meta device context.
        _orig_get_init_context = MultiModalityCausalLM.get_init_context

        @classmethod
        def _no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            contexts = _orig_get_init_context.__func__(
                cls, dtype, is_quantized, _is_ds_init_called
            )
            return [c for c in contexts if not isinstance(c, torch.device)]

        MultiModalityCausalLM.get_init_context = _no_meta_get_init_context
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_name), **model_kwargs
            )
        finally:
            MultiModalityCausalLM.get_init_context = _orig_get_init_context
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Janus."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        inputs = self.processor(
            images=image, text=self.sample_text, return_tensors="pt"
        )

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs
