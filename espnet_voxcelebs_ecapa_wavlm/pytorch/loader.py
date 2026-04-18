# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ESPnet ECAPA-TDNN WavLM joint speaker embedding model loader.
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
    """Available ESPnet ECAPA-TDNN WavLM model variants."""

    ECAPA_WAVLM_JOINT = "ECAPA_WavLM_Joint"


class ModelLoader(ForgeModel):
    """ESPnet ECAPA-TDNN WavLM joint speaker embedding model loader."""

    _VARIANTS = {
        ModelVariant.ECAPA_WAVLM_JOINT: ModelConfig(
            pretrained_model_name="espnet/voxcelebs12_ecapa_wavlm_joint",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ECAPA_WAVLM_JOINT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ESPnetECAPAWavLMVoxCeleb",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_torchaudio_compat():
        import sys
        import types

        import torchaudio

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda x: None

        # s3prl expects torchaudio.sox_effects which was removed in torchaudio 2.1+
        if "torchaudio.sox_effects" not in sys.modules:
            stub = types.ModuleType("torchaudio.sox_effects")
            stub.apply_effects_tensor = lambda *a, **kw: None
            sys.modules["torchaudio.sox_effects"] = stub

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ESPnet ECAPA-TDNN WavLM joint speaker embedding model."""
        self._patch_torchaudio_compat()

        from espnet2.bin.spk_inference import Speech2Embedding

        speech2embed = Speech2Embedding.from_pretrained(
            model_tag=self._variant_config.pretrained_model_name, **kwargs
        )
        spk_model = speech2embed.spk_model

        # The s3prl/WavLM frontend uses list comprehensions and
        # data-dependent shapes that are incompatible with torch.compile.
        # Extract the downstream components (normalize, encoder, pooling,
        # projector) into a compile-friendly wrapper that takes
        # pre-extracted frontend features as input.
        class PostFrontendWrapper(torch.nn.Module):
            def __init__(self, spk):
                super().__init__()
                self.normalize = spk.normalize
                self.encoder = spk.encoder
                self.pooling = spk.pooling
                self.projector = spk.projector

            def forward(self, feats):
                feats, _ = self.normalize(feats, None)
                frame_feats = self.encoder(feats)
                utt_feat = self.pooling(frame_feats, feat_lengths=None)
                embd = self.projector(utt_feat)
                return embd

        model = PostFrontendWrapper(spk_model)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample frontend features for the ECAPA-TDNN encoder.

        Returns a tensor matching the WavLM frontend output shape
        for 1 second of 16kHz audio: (batch, 50 frames, 1024 features).
        """
        feats = torch.randn(1, 50, 1024)

        if dtype_override is not None:
            feats = feats.to(dtype_override)

        return [feats]
