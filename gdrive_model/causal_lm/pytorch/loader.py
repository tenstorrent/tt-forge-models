# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Google Drive model loader for loading model weights from your personal Google Drive.

This loader demonstrates how to:
1. Load model weights from YOUR Google Drive
2. Use cache-first logic (second run uses cached weights)

Setup:
    1. Save model weights locally:
       python third_party/tt_forge_models/tools/gdrive_upload_example.py --save --source-model "gpt2"
    2. Upload the weights file to Google Drive manually
    3. Get the shareable link and extract the file ID
    4. Set environment variables:
       export GDRIVE_FILE_ID="your-file-id"
       export GDRIVE_FILENAME="gpt2_weights.pt"

Usage:
    export GDRIVE_FILE_ID="1ABC123xyz"
    export GDRIVE_FILENAME="gpt2_weights.pt"
    pytest -svv tests/runner/test_models.py::test_all_models_torch[gdrive_model/causal_lm/pytorch-gdrive-single_device-inference]
"""
import os
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
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
from ....tools.utils import download_from_gdrive, is_gdrive_file_cached
import torch


def get_gdrive_config() -> tuple:
    """Get Google Drive configuration from environment variables.
    
    Returns:
        tuple: (file_id, filename)
    
    Raises:
        ValueError: If required environment variables are not set
    """
    file_id = os.environ.get("GDRIVE_FILE_ID")
    filename = os.environ.get("GDRIVE_FILENAME", "model_weights.pt")
    
    if not file_id:
        raise ValueError(
            "GDRIVE_FILE_ID environment variable is not set.\n"
            "Please set it to your Google Drive file ID:\n"
            '  export GDRIVE_FILE_ID="your-google-drive-file-id"\n'
            "\nTo get the file ID:\n"
            "  1. Upload weights to Google Drive\n"
            "  2. Right-click -> Share -> Anyone with link\n"
            "  3. Copy link: https://drive.google.com/file/d/FILE_ID/view\n"
            "  4. The FILE_ID is the string between /d/ and /view"
        )
    return file_id, filename


class ModelVariant(StrEnum):
    """Available model variants."""
    
    GDRIVE = "gdrive"


class ModelLoader(ForgeModel):
    """Google Drive model loader with cache-first logic.
    
    This loader reads the file ID from GDRIVE_FILE_ID environment variable
    and downloads/caches the model weights from Google Drive.
    
    First run: Downloads from Google Drive
    Second run: Uses cached weights (no download needed)
    """

    # Dictionary of available model variants - required for test discovery
    _VARIANTS = {
        ModelVariant.GDRIVE: LLMModelConfig(
            pretrained_model_name="gdrive",  # Placeholder
            max_length=128,
        ),
    }

    # Default variant
    DEFAULT_VARIANT = ModelVariant.GDRIVE

    # Sample text for testing
    sample_text = "The quick brown fox jumps over the lazy"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader."""
        super().__init__(variant)
        self.tokenizer = None
        self._file_id = None
        self._filename = None

    def _get_gdrive_config(self) -> tuple:
        """Get and cache the Google Drive config."""
        if self._file_id is None:
            self._file_id, self._filename = get_gdrive_config()
        return self._file_id, self._filename

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model info."""
        return ModelInfo(
            model="gdrive_model",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load GPT-2 tokenizer (used for the sample model)."""
        # Using GPT-2 tokenizer as default since we're loading GPT-2 weights
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load model weights from Google Drive with cache-first logic.

        First run: Downloads from Google Drive
        Second run: Uses cached weights

        Args:
            dtype_override: Optional torch.dtype to override model dtype

        Returns:
            torch.nn.Module: The loaded model
        """
        file_id, filename = self._get_gdrive_config()
        
        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Check cache status
        is_cached = is_gdrive_file_cached(file_id, filename)
        print(f"\n[GDrive Model] File ID: {file_id}", flush=True)
        print(f"[GDrive Model] Cache status: {'CACHED ✓' if is_cached else 'NOT CACHED'}", flush=True)

        # Download or load from cache
        weights_path = download_from_gdrive(file_id, filename)

        # Create model architecture (GPT-2 as example)
        config = GPT2Config()
        model = GPT2LMHeadModel(config)

        # Load weights
        print(f"[GDrive Model] Loading weights from {weights_path}...", flush=True)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        # Apply dtype override
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        print(f"[GDrive Model] Model loaded successfully", flush=True)
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the model."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs to text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits.argmax(dim=-1)
        predicted_text = self.tokenizer.decode(
            predicted_token_ids[0], skip_special_tokens=True
        )

        return predicted_text

