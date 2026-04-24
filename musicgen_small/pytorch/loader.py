# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Musicgen-small model loader implementation
"""
from typing import Optional

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

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
    """Available Musicgen-small model variants."""

    FACEBOOK = "facebook/musicgen-small"
    XENOVA = "Xenova/musicgen-small"


class ModelLoader(ForgeModel):
    """Musicgen-small model loader implementation."""

    _VARIANTS = {
        ModelVariant.FACEBOOK: ModelConfig(
            pretrained_model_name="facebook/musicgen-small",
        ),
        ModelVariant.XENOVA: ModelConfig(
            pretrained_model_name="facebook/musicgen-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FACEBOOK

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant. If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MusicGen Small",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the Musicgen-small model instance with default settings.

        Returns:
            torch.nn.Module: The Musicgen-small model instance.

        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name, **kwargs)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            pretrained_model_name, **kwargs
        )
        return self.model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Musicgen-small model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self.load_model()

        inputs = self.processor(
            text=[
                "80s pop track with bassy drums and synth",
                "90s rock song with loud guitars and heavy drums",
            ],
            padding=True,
            return_tensors="pt",
        )

        # If batch_size is different from 2, adjust using repeat_interleave
        if batch_size != 2:
            # Calculate how many times to repeat each example
            repeats_per_example = batch_size // 2
            remaining = batch_size % 2

            # Apply repeat_interleave to input tensors
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    if remaining == 0:
                        # Even division
                        inputs[key] = inputs[key].repeat_interleave(
                            repeats_per_example, dim=0
                        )
                    else:
                        # Handle remainder by repeating first example one extra time
                        repeated = inputs[key].repeat_interleave(
                            repeats_per_example, dim=0
                        )
                        extra = inputs[key][:1].repeat_interleave(remaining, dim=0)
                        inputs[key] = torch.cat([repeated, extra], dim=0)

        pad_token_id = self.model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones(
                (
                    inputs.input_ids.shape[0] * self.model.decoder.num_codebooks,
                    1,
                ),
                dtype=torch.long,
            )
            * pad_token_id
        )

        inputs["max_new_tokens"] = 1
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
