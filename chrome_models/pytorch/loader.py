# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chrome on-device models loader implementation.

dejanseo/chrome_models is a collection of Google Chrome's on-device machine
learning models distributed as TensorFlow Lite files. Each model corresponds
to a Chrome "optimization target" such as LANGUAGE_DETECTION, PAGE_TOPICS_V2
or PAGE_VISIBILITY.

The original TFLite models use the NGramHash custom op (from the TFLite Support
Library) which is not available in standard TFLite/ai-edge-litert packages, and
takes string tensors as input — both of which are incompatible with
PyTorch/XLA compilation.

This loader provides a pure PyTorch structural approximation of the language
detection architecture. It accepts a byte-encoded text sequence (int64 tensor)
and replicates the FC-layer structure of the original TFLite model.
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

# Fixed input length: 512 byte values representing encoded text
_INPUT_SEQ_LEN = 512
# Number of supported languages in the original Chrome model
_NUM_LANGUAGES = 111


class ModelVariant(StrEnum):
    """Available Chrome on-device model variants."""

    LANGUAGE_DETECTION = "language_detection"


class ChromeLanguageDetectorProxy(nn.Module):
    """Pure PyTorch proxy for Chrome's on-device language detection model.

    Replicates the FC-layer architecture of the original TFLite model while
    remaining compatible with PyTorch/XLA compilation. The original model uses
    NGramHash (unavailable custom op) and string tensors; this proxy accepts
    int64 byte values instead and uses a learned byte embedding in place of the
    N-gram hash feature extraction pipeline.

    Input:  int64 tensor of shape [_INPUT_SEQ_LEN] — text encoded as byte values
    Output: float32 tensor of shape [_NUM_LANGUAGES] — language probabilities
    """

    def __init__(self):
        super().__init__()
        # Byte-level embedding to replace the NGramHash + EmbeddingLookup pipeline.
        # Vocab size 256 (byte values 0-255), embedding dim 64 to match the
        # original model's FC1 input width (pod/fully_connected weight: [250, 64]).
        self.byte_embedding = nn.Embedding(256, 64)
        # FC layers matching the original TFLite architecture sizes.
        self.fc1 = nn.Linear(64, 250)
        self.fc2 = nn.Linear(250, 200)
        self.fc3 = nn.Linear(200, _NUM_LANGUAGES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [_INPUT_SEQ_LEN] int64 byte values (clamped to [0, 255])
        x = x.clamp(0, 255)
        emb = self.byte_embedding(x)  # [_INPUT_SEQ_LEN, 64]
        pooled = emb.mean(dim=0)  # [64]
        h1 = torch.relu(self.fc1(pooled))  # [250]
        h2 = torch.relu(self.fc2(h1))  # [200]
        out = self.fc3(h2)  # [_NUM_LANGUAGES]
        return torch.softmax(out, dim=-1)


class ModelLoader(ForgeModel):
    """Loader for Chrome language detection model (pure PyTorch proxy)."""

    _VARIANTS = {
        ModelVariant.LANGUAGE_DETECTION: ModelConfig(
            pretrained_model_name="dejanseo/chrome_models",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LANGUAGE_DETECTION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="chrome_models",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_model(self):
        if self._model is None:
            self._model = ChromeLanguageDetectorProxy()
            self._model.eval()
        return self._model

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the Chrome language detection proxy model."""
        return self._ensure_model()

    def load_inputs(self, dtype_override=None):
        """Return a sample input: byte-encoded text of fixed length."""
        return [torch.randint(0, 256, (_INPUT_SEQ_LEN,), dtype=torch.int64)]
