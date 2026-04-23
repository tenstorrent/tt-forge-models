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


def _patch_asp_for_dtype(model):
    """Keep AttentiveStatisticsPooling in float32 and wrap it to cast in/out.

    SpeechBrain's ASP forward explicitly calls .float() on a mask tensor, which
    promotes intermediate tensors to float32 even when the model is bfloat16.
    We keep ASP and its sub-modules in float32 and wrap the forward to cast
    the bfloat16 input to float32 on entry and back to bfloat16 on exit.
    """
    import types

    for module in model.modules():
        if module.__class__.__name__ != "AttentiveStatisticsPooling":
            continue
        module.float()
        orig = module.forward.__func__

        def _make_patched(fn):
            def patched(self, x, lengths=None):
                return fn(self, x.float(), lengths).to(x.dtype)

            return patched

        module.forward = types.MethodType(_make_patched(orig), module)


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

        if dtype_override is not None:
            model = model.to(dtype_override)
            # SpeechBrain's AttentiveStatisticsPooling uses .float() internally,
            # causing float32/bfloat16 mismatches. Patch ASP layers to run in
            # float32 and cast the output back to the target dtype.
            _patch_asp_for_dtype(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample Fbank feature input for the ECAPA-TDNN embedding model.

        Returns pre-computed features of shape (batch, time_steps, n_mels)
        equivalent to 1 second of 16kHz audio processed through Fbank features.
        """
        features = torch.randn(1, 101, 80)

        if dtype_override is not None:
            features = features.to(dtype_override)

        return [features]
