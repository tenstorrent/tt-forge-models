# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
griko/gender_cls_svm_ecapa_voxceleb gender classification model loader.

The upstream pipeline stacks an SVM gender classifier on top of SpeechBrain's
speechbrain/spkrec-ecapa-voxceleb ECAPA-TDNN embedding model. Only the
ECAPA-TDNN PyTorch backbone is exercised here; the scikit-learn SVM head is
not part of the PyTorch graph.
"""

from typing import Optional

import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available griko gender classification model variants."""

    GRIKO_GENDER_CLS_SVM_ECAPA_VOXCELEB = "griko_gender_cls_svm_ecapa_voxceleb"


class ModelLoader(ForgeModel):
    """griko/gender_cls_svm_ecapa_voxceleb ECAPA-TDNN embedding backbone loader."""

    _VARIANTS = {
        ModelVariant.GRIKO_GENDER_CLS_SVM_ECAPA_VOXCELEB: ModelConfig(
            pretrained_model_name="speechbrain/spkrec-ecapa-voxceleb",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRIKO_GENDER_CLS_SVM_ECAPA_VOXCELEB

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GenderClsSvmEcapaVoxceleb",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SpeechBrain ECAPA-TDNN embedding backbone."""
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source=self._variant_config.pretrained_model_name, **kwargs
        )
        model = classifier.mods.embedding_model
        model.eval()

        # SpeechBrain's BatchNorm1d layers internally promote bfloat16 to float32
        # causing dtype mismatches with bfloat16 conv bias on CPU. Keep float32.

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample Fbank feature input for the ECAPA-TDNN embedding model.

        Returns pre-computed features of shape (batch, time_steps, n_mels)
        equivalent to 1 second of 16kHz audio processed through Fbank features.
        """
        return [torch.randn(1, 101, 80)]
