# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image text-encoder loader for text-to-image generation.

Qwen-Image conditions the denoiser on the last-layer hidden states of a
``Qwen2_5_VLForConditionalGeneration`` model (used text-only here). A thin
wrapper exposes those hidden states as ``forward`` output so the generic
single-forward-pass test harness exercises the encoder.
"""
import torch
from typing import Optional

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

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


class ModelVariant(StrEnum):
    """Available Qwen-Image text-encoder variants."""

    BASE = "Base"


class _QwenImageTextEncoder(torch.nn.Module):
    """Wraps the VL model so forward() returns last-layer prompt hidden states."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]


class ModelLoader(ForgeModel):
    """Qwen-Image text-encoder loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Qwen/Qwen-Image",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Representative prompt sequence length.
    SEQ_LEN = 256

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.text_encoder = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="qwen_image_text_encoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, subfolder="tokenizer"
            )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Qwen-Image text-encoder wrapper.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype
                            (weights are distributed in bfloat16).

        Returns:
            torch.nn.Module: Wrapper returning last-layer hidden states.
        """
        model_kwargs = {"subfolder": "text_encoder", "use_safetensors": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        text_encoder = text_encoder.eval()
        if dtype_override is not None:
            text_encoder = text_encoder.to(dtype_override)

        self.text_encoder = text_encoder
        return _QwenImageTextEncoder(text_encoder)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load tokenized prompt inputs for the Qwen-Image text encoder.

        Args:
            dtype_override: Unused (integer token ids).
            batch_size: Optional batch size (default 1).

        Returns:
            dict: {"input_ids", "attention_mask"} padded to SEQ_LEN.
        """
        self._load_tokenizer()

        prompt = (
            "A coffee shop entrance features a chalkboard sign reading "
            '"Qwen Coffee", with a neon sign in the window.'
        )
        tokens = self.tokenizer(
            [prompt] * batch_size,
            max_length=self.SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
        }
