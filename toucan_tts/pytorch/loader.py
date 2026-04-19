# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys

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

IMS_TOUCAN_REPO = "https://github.com/DigitalPhonetics/IMS-Toucan.git"
IMS_TOUCAN_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "ims_toucan", "IMS-Toucan"
)


def _ensure_ims_toucan():
    if not os.path.isdir(IMS_TOUCAN_DIR):
        os.makedirs(os.path.dirname(IMS_TOUCAN_DIR), exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--filter=blob:none", IMS_TOUCAN_REPO, IMS_TOUCAN_DIR]
        )
    if IMS_TOUCAN_DIR not in sys.path:
        sys.path.insert(0, IMS_TOUCAN_DIR)
    # Stub out heavy transitive deps that are imported at module level
    # but never exercised during model construction/inference.
    from unittest.mock import MagicMock

    for mod_name in [
        "dragonmapper",
        "dragonmapper.transcriptions",
        "phonemizer",
        "phonemizer.backend",
        "pypinyin",
        "speechbrain",
        "speechbrain.pretrained",
        "sounddevice",
        "pyloudnorm",
        "pyloudnorm.normalize",
        "torchaudio",
        "torchaudio.transforms",
        "librosa",
        "epitran",
        "g2pk",
        "jamo",
        "pykakasi",
        "transphone",
        "transphone.g2p",
        "phonepiece",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


def _patch_ims_toucan():
    import functools

    from Modules.ToucanTTS import flow_matching, dit
    from Modules.GeneralLayers import PositionalEncoding

    # Remove @torch.inference_mode() from CFMDecoder.forward
    # (conflicts with TT custom ops that need to set version_counter)
    orig_cfm_fwd = flow_matching.CFMDecoder.forward.__wrapped__
    flow_matching.CFMDecoder.forward = orig_cfm_fwd

    # Patch RotaryPositionalEmbeddings._build_cache to always rebuild
    # (prevents device mismatch between cached CPU tensors and XLA tensors)
    orig_build_cache = dit.RotaryPositionalEmbeddings._build_cache

    @functools.wraps(orig_build_cache)
    def _build_cache_no_skip(self, x):
        self.cos_cached = None
        orig_build_cache(self, x)

    dit.RotaryPositionalEmbeddings._build_cache = _build_cache_no_skip

    # Patch RelPositionalEncoding.extend_pe to always rebuild
    # (cached PE on CPU causes shape issues during XLA FakeTensor tracing)
    orig_extend_pe = PositionalEncoding.RelPositionalEncoding.extend_pe

    @functools.wraps(orig_extend_pe)
    def _extend_pe_no_skip(self, x):
        self.pe = None
        orig_extend_pe(self, x)

    PositionalEncoding.RelPositionalEncoding.extend_pe = _extend_pe_no_skip


def _scale_variance_safe(sequence, scale):
    if scale == 1.0:
        return sequence
    average = sequence[0][sequence[0] != 0.0].mean()
    sequence = sequence - average
    sequence = sequence * scale
    sequence = sequence + average
    sequence = torch.clamp(sequence, min=0.0)
    return sequence


def _forward_no_inplace(
    model, text_tensors, text_lengths, utterance_embedding=None, lang_ids=None
):
    from Preprocessing.articulatory_features import get_feature_to_index_lookup

    text_tensors = torch.clamp(text_tensors, max=1.0)

    if not model.multilingual_model:
        lang_ids = None

    if not model.multispeaker_model:
        utterance_embedding = None

    if utterance_embedding is not None:
        utterance_embedding = torch.nn.functional.normalize(utterance_embedding)
        if model.integrate_language_embedding_into_encoder_out and lang_ids is not None:
            lang_embs = model.encoder.language_embedding(lang_ids)
            lang_embs = torch.nn.functional.normalize(lang_embs)
            utterance_embedding = torch.cat(
                [lang_embs, utterance_embedding], dim=1
            ).detach()

    seq_len = text_tensors.shape[1]
    ones_mask = torch.ones(
        1, 1, seq_len, dtype=text_tensors.dtype, device=text_tensors.device
    )

    encoded_texts, _ = model.encoder(
        text_tensors,
        ones_mask,
        utterance_embedding=utterance_embedding,
        lang_ids=lang_ids,
    )

    reduced_pitch_space = model.pitch_latent_reduction(encoded_texts).transpose(1, 2)
    pitch_predictions = model.pitch_predictor(
        mu=reduced_pitch_space,
        mask=ones_mask,
        n_timesteps=20,
        temperature=0.1,
        c=utterance_embedding,
    )
    pitch_predictions = pitch_predictions.clone()
    pitch_predictions[0][0][0] = pitch_predictions[0][0][1]
    pitch_predictions[0][0][-1] = pitch_predictions[0][0][-3]
    pitch_predictions[0][0][-2] = pitch_predictions[0][0][-3]
    pitch_predictions = _scale_variance_safe(pitch_predictions, 1.0)
    embedded_pitch_curve = model.pitch_embed(pitch_predictions).transpose(1, 2)

    reduced_energy_space = model.energy_latent_reduction(
        encoded_texts + embedded_pitch_curve
    ).transpose(1, 2)
    energy_predictions = model.energy_predictor(
        mu=reduced_energy_space,
        mask=ones_mask,
        n_timesteps=20,
        temperature=0.1,
        c=utterance_embedding,
    )
    energy_predictions = energy_predictions.clone()
    energy_predictions[0][0][0] = energy_predictions[0][0][1]
    energy_predictions[0][0][-1] = energy_predictions[0][0][-3]
    energy_predictions[0][0][-2] = energy_predictions[0][0][-3]
    energy_predictions = _scale_variance_safe(energy_predictions, 1.0)
    embedded_energy_curve = model.energy_embed(energy_predictions).transpose(1, 2)

    reduced_duration_space = model.duration_latent_reduction(
        encoded_texts + embedded_pitch_curve + embedded_energy_curve
    ).transpose(1, 2)
    predicted_durations = (
        torch.clamp(
            torch.ceil(
                model.duration_predictor(
                    mu=reduced_duration_space,
                    mask=ones_mask,
                    n_timesteps=20,
                    temperature=0.1,
                    c=utterance_embedding,
                )
            ),
            min=0.0,
        )
        .long()
        .squeeze(1)
    )

    predicted_durations = predicted_durations.clone()
    predicted_durations[0][0] = 1
    wb_idx = get_feature_to_index_lookup()["word-boundary"]
    for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
        if phoneme_vector[wb_idx] == 1:
            predicted_durations[0][phoneme_index] = 0

    enriched_encoded_texts = (
        encoded_texts + embedded_pitch_curve + embedded_energy_curve
    )
    upsampled_enriched_encoded_texts = model.length_regulator(
        enriched_encoded_texts, predicted_durations
    )

    decoded_speech, _ = model.decoder(
        upsampled_enriched_encoded_texts, None, utterance_embedding=utterance_embedding
    )
    preliminary_spectrogram = model.output_projection(decoded_speech)

    dec_len = decoded_speech.shape[1]
    dec_mask = torch.ones(
        1, 1, dec_len, dtype=decoded_speech.dtype, device=decoded_speech.device
    )
    refined_codec_frames = model.flow_matching_decoder(
        mu=preliminary_spectrogram.transpose(1, 2),
        mask=dec_mask,
        n_timesteps=30,
        temperature=0.2,
        c=None,
    ).transpose(1, 2)

    return (
        refined_codec_frames,
        predicted_durations.squeeze(),
        pitch_predictions.squeeze(),
        energy_predictions.squeeze(),
    )


class ToucanTTSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, text_lengths, utterance_embedding, lang_ids):
        return _forward_no_inplace(
            self.model,
            text_tensors=text,
            text_lengths=text_lengths,
            utterance_embedding=utterance_embedding,
            lang_ids=lang_ids,
        )


class ModelVariant(StrEnum):
    MULTILINGUAL = "Multilingual"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.MULTILINGUAL: ModelConfig(
            pretrained_model_name="Flux9665/ToucanTTS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTILINGUAL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ToucanTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_ims_toucan()
        _patch_ims_toucan()
        from huggingface_hub import hf_hub_download
        from Modules.ToucanTTS.InferenceToucanTTS import ToucanTTS

        checkpoint_path = hf_hub_download(
            repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = ToucanTTS(weights=checkpoint["model"], config=checkpoint["config"])
        with torch.no_grad():
            model.store_inverse_all()
        wrapper = ToucanTTSWrapper(model)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        dtype = torch.float32
        batch = 1
        text_len = 32
        phone_feature_dim = 64

        text = torch.randn(batch, text_len, phone_feature_dim, dtype=dtype)
        text_lengths = torch.tensor([text_len], dtype=torch.long)
        utterance_embedding = torch.randn(batch, 192, dtype=dtype)
        lang_ids = torch.tensor([0], dtype=torch.long)

        return text, text_lengths, utterance_embedding, lang_ids
