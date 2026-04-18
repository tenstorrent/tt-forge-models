# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA Streaming Sortformer speaker diarization model loader implementation using PyTorch.

Loads the SortformerEncLabelModel for end-to-end speaker diarization,
identifying and labeling individual speakers in multi-speaker audio.
"""

import torch
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


class SortformerEncoderWrapper(torch.nn.Module):
    """Wraps the Sortformer encoder and transformer, bypassing
    the STFT-based audio preprocessor which produces complex-valued
    tensors unsupported by the XLA backend."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.sortformer_modules = model.sortformer_modules
        self.transformer_encoder = model.transformer_encoder

    def forward(self, processed_signal, processed_signal_length):
        emb_seq, emb_seq_length = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
        )
        emb_seq = emb_seq.transpose(1, 2)
        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)
        encoder_mask = self.sortformer_modules.length_to_mask(
            emb_seq_length, emb_seq.shape[1]
        )
        trans_emb_seq = self.transformer_encoder(
            encoder_states=emb_seq, encoder_mask=encoder_mask
        )
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
        preds = preds * encoder_mask.unsqueeze(-1)
        return preds


class ModelVariant(StrEnum):
    """Available Sortformer speaker diarization model variants."""

    STREAMING_4SPK_V2_1 = "Streaming_4spk_v2.1"


class ModelLoader(ForgeModel):
    """NVIDIA Streaming Sortformer speaker diarization model loader (PyTorch)."""

    _VARIANTS = {
        ModelVariant.STREAMING_4SPK_V2_1: ModelConfig(
            pretrained_model_name="nvidia/diar_streaming_sortformer_4spk-v2.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STREAMING_4SPK_V2_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Sortformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from nemo.collections.asr.models import SortformerEncLabelModel

        model = SortformerEncLabelModel.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model.eval()

        wrapper = SortformerEncoderWrapper(model)
        wrapper.eval()

        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)

        return wrapper

    def load_inputs(self, dtype_override=None):
        from nemo.collections.asr.models import SortformerEncLabelModel

        model = SortformerEncLabelModel.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model.eval()

        sampling_rate = 16000
        audio_signal = torch.randn(1, sampling_rate)
        audio_signal_length = torch.tensor([sampling_rate])

        with torch.no_grad():
            processed_signal, processed_signal_length = model.process_signal(
                audio_signal=audio_signal, audio_signal_length=audio_signal_length
            )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]

        if dtype_override is not None:
            processed_signal = processed_signal.to(dtype_override)

        return {
            "processed_signal": processed_signal,
            "processed_signal_length": processed_signal_length,
        }
