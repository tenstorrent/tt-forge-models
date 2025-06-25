# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper model loader implementation
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from ...base import ForgeModel
from datasets import load_dataset


class ModelLoader(ForgeModel):
    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "openai/whisper-tiny"
        self.processor = None

    def load_model(self, dtype_override=None):
        """Load a Whisper model from Hugging Face."""

        # Initialize processor first with default or overridden dtype
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, use_cache=False, return_dict=False, **processor_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, use_cache=False, return_dict=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self):
        """Generate sample inputs for Whisper model."""

        # Ensure processor is initialized
        if not hasattr(cls, "processor"):
            self.load_model()  # This will initialize the processor

        # load dummy dataset and read audio files
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample = ds[0]["audio"]
        inputs = self.processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features
        return inputs
