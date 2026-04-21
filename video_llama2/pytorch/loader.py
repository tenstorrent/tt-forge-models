# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoLLaMA2 model loader implementation for multimodal audio-visual video understanding.
"""

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available VideoLLaMA2 model variants."""

    V2_1_7B_AV = "v2_1_7b_av"


class ModelLoader(ForgeModel):
    """VideoLLaMA2 model loader for multimodal audio-visual video understanding."""

    _VARIANTS = {
        ModelVariant.V2_1_7B_AV: ModelConfig(
            pretrained_model_name="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_1_7B_AV

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoLLaMA2 model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoLLaMA2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoLLaMA2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        kwargs.setdefault("trust_remote_code", True)
        model = AutoModelForCausalLM.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoLLaMA2."""
        if self.tokenizer is None:
            self._load_tokenizer()

        text_prompt = "<video>\nPlease describe the video with audio information."

        inputs = self.tokenizer(text_prompt, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
