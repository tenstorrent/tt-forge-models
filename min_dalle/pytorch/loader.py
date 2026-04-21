# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
min(DALL-E) model loader implementation for text-to-image generation.

Loads the DalleBartEncoder from kuprel/min-dalle, a minimal PyTorch port of
Boris Dayma's DALL-E Mini/Mega model. Weights are pulled from the Hugging
Face repo via the ``min_dalle`` package.
"""

import tempfile
from typing import Optional

import torch

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
    """Available min(DALL-E) variants."""

    MINI = "mini"
    MEGA = "mega"


class ModelLoader(ForgeModel):
    """min(DALL-E) model loader implementation."""

    _VARIANTS = {
        ModelVariant.MINI: ModelConfig(
            pretrained_model_name="kuprel/min-dalle",
        ),
        ModelVariant.MEGA: ModelConfig(
            pretrained_model_name="kuprel/min-dalle",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI

    sample_text = "Nuclear explosion broccoli"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._min_dalle = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="min-DALL-E",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _init_pipeline(self, dtype: torch.dtype):
        from min_dalle import MinDalle

        # is_reusable=False so we can initialize only the encoder below,
        # avoiding the (mega) decoder / VQ-GAN detokenizer downloads we
        # don't exercise in this encoder-only test.
        self._min_dalle = MinDalle(
            models_root=tempfile.mkdtemp(prefix="min_dalle_"),
            dtype=dtype,
            device="cpu",
            is_mega=self._variant == ModelVariant.MEGA,
            is_reusable=False,
            is_verbose=False,
        )
        return self._min_dalle

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the DalleBartEncoder with weights from kuprel/min-dalle."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._min_dalle is None:
            self._init_pipeline(dtype)

        self._min_dalle.init_encoder()
        return self._min_dalle.encoder.eval()

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Prepare tokenized text inputs for the DalleBartEncoder."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._min_dalle is None:
            self._init_pipeline(dtype)

        tokens = self._min_dalle.tokenizer.tokenize(self.sample_text)
        if len(tokens) > self._min_dalle.text_token_count:
            tokens = tokens[: self._min_dalle.text_token_count]

        # DALL-E Bart conditions on two sequences: a short [BOS, EOS] row and
        # the full tokenized prompt. Positions past the prompt length are set
        # to the padding token (1), matching the encoder's attention mask.
        text_tokens = torch.ones(
            (2, self._min_dalle.text_token_count), dtype=torch.long
        )
        text_tokens[0, :2] = torch.tensor([tokens[0], tokens[-1]])
        text_tokens[1, : len(tokens)] = torch.tensor(tokens)

        return [text_tokens]
