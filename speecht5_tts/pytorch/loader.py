# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechT5 TTS model loader implementation
"""

import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="speecht5_tts",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "microsoft/speecht5_tts"
        self.processor = None

    def load_model(self, dtype_override=None):
        """Load a SpeechT5 TTS model from Hugging Face."""
        dtype = dtype_override or torch.bfloat16

        # Initialize processor
        self.processor = SpeechT5Processor.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model = SpeechT5ForTextToSpeech.from_pretrained(
            self.model_name, torch_dtype=dtype
        )

        # Return the generate_speech method as in the original test
        return model.generate_speech

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for SpeechT5 TTS model."""
        dtype = dtype_override or torch.bfloat16

        # Create tokenized inputs
        inputs = self.processor(text="Hello, my dog is cute.", return_tensors="pt")

        # Load speaker embeddings (zeros as in original test)
        speaker_embeddings = torch.zeros((1, 512)).to(dtype)

        # Load vocoder
        vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan", torch_dtype=dtype
        )

        arguments = {
            "input_ids": inputs["input_ids"],
            "speaker_embeddings": speaker_embeddings,
            "vocoder": vocoder,
        }

        return arguments
