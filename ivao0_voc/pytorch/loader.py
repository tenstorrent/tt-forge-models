# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ivao0/voc Attentionless Vocoder Streaming loader implementation.

Voc is a neural audio tokenizer/vocoder based on kyutai/mimi that both
encodes 24 kHz audio into discrete tokens and decodes tokens back into
audio.  This loader exposes the decode path so a forward pass converts
codebook tokens into an audio waveform.
"""
import torch
import torch.nn as nn
from typing import Optional

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
from .src.model import Voc


class VocDecodeWrapper(nn.Module):
    """Single-step decode wrapper around Voc.

    The native Voc.decode iterates over the time axis of the codes tensor
    and relies on streaming state inside its buffered convolutions.  For a
    clean, traceable forward pass we invoke the per-frame decode pipeline
    once on a single timestep of codes.
    """

    def __init__(self, model):
        super().__init__()
        self.quantizer = model.quantizer
        self.upsample = model.upsample
        self.decoder_transformer = model.decoder_transformer
        self.decoder = model.decoder

    def forward(self, codes):
        x = self.quantizer.decode(codes)
        x = self.upsample(x)
        x = self.decoder_transformer(x)
        return self.decoder(x)


class ModelVariant(StrEnum):
    """Available ivao0/voc model variants."""

    VOC = "voc"


class ModelLoader(ForgeModel):
    """ivao0/voc Attentionless Vocoder Streaming loader implementation."""

    _VARIANTS = {
        ModelVariant.VOC: ModelConfig(
            pretrained_model_name="ivao0/voc",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOC

    # The quantizer emits 1 semantic codebook + 21 acoustic codebook
    # selections per audio frame.
    NUM_CODEBOOKS = 22
    CODEBOOK_SIZE = 2048

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ivao0_voc",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Voc model wrapped for decode inference.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: Wrapped Voc model that decodes codes to audio.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        voc = Voc.from_pretrained(pretrained_model_name)

        if dtype_override is not None:
            voc = voc.to(dtype=dtype_override)

        model = VocDecodeWrapper(voc)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Voc decode wrapper.

        Generates a single timestep of codebook indices (semantic + acoustic)
        suitable for the per-frame decode pipeline.

        Args:
            dtype_override: Unused; codebook indices are always integer-valued.

        Returns:
            torch.Tensor: Code tensor of shape (1, 22, 1).
        """
        codes = torch.randint(
            low=0,
            high=self.CODEBOOK_SIZE,
            size=(1, self.NUM_CODEBOOKS, 1),
            dtype=torch.long,
        )
        return codes
