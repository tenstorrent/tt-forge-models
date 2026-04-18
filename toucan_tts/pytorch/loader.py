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
IMS_TOUCAN_COMMIT = "3cc2094d9c7123336eda7e299ac0bc90319ca9ff"


def _patch_file(path, replacements):
    with open(path, "r") as f:
        src = f.read()
    for old, new in replacements:
        src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)


def _patch_ims_toucan(cache_dir):
    tts_path = os.path.join(cache_dir, "Modules", "ToucanTTS", "InferenceToucanTTS.py")
    _patch_file(
        tts_path,
        [
            (
                "pitch_predictions[0][0][0] = pitch_predictions[0][0][1]",
                "pitch_predictions = pitch_predictions.clone()\n"
                "        pitch_predictions[0][0][0] = pitch_predictions[0][0][1]",
            ),
            (
                "energy_predictions[0][0][0] = energy_predictions[0][0][1]",
                "energy_predictions = energy_predictions.clone()\n"
                "        energy_predictions[0][0][0] = energy_predictions[0][0][1]",
            ),
            (
                "predicted_durations[0][0] = 1",
                "predicted_durations = predicted_durations.clone()\n"
                "        predicted_durations[0][0] = 1",
            ),
            (
                "            sequence[0][0][sequence_index] = 0.0",
                "            sequence = sequence.clone()\n"
                "            sequence[0][0][sequence_index] = 0.0",
            ),
        ],
    )

    cfm_path = os.path.join(cache_dir, "Modules", "ToucanTTS", "flow_matching.py")
    _patch_file(
        cfm_path,
        [
            ("    @torch.inference_mode()\n    def forward(", "    def forward("),
        ],
    )

    attn_path = os.path.join(cache_dir, "Modules", "GeneralLayers", "Attention.py")
    _patch_file(
        attn_path,
        [
            (
                "min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)",
                "min_value = torch.finfo(scores.dtype).min",
            ),
        ],
    )

    dit_path = os.path.join(cache_dir, "Modules", "ToucanTTS", "dit.py")
    _patch_file(
        dit_path,
        [
            (
                "        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:\n"
                "            return",
                "        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0] and self.cos_cached.device == x.device:\n"
                "            return",
            ),
        ],
    )

    tts_fwd_path = os.path.join(
        cache_dir, "Modules", "ToucanTTS", "InferenceToucanTTS.py"
    )
    _patch_file(
        tts_fwd_path,
        [
            (
                "    def _forward(self,\n"
                "                 text_tensors,\n"
                "                 text_lengths,",
                "    def _forward(self,\n"
                "                 text_tensors,\n"
                "                 text_lengths,\n"
                "                 text_masks=None,",
            ),
            (
                "        text_masks = make_non_pad_mask(text_lengths, device=text_lengths.device).unsqueeze(-2)",
                "        if text_masks is None:\n"
                "            text_masks = make_non_pad_mask(text_lengths, device=text_lengths.device).unsqueeze(-2)",
            ),
        ],
    )


def _ensure_ims_toucan():
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "IMS-Toucan")
    marker = os.path.join(cache_dir, ".patched")
    if not os.path.isdir(cache_dir):
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--filter=blob:none", "-q", IMS_TOUCAN_REPO, cache_dir]
        )
        subprocess.check_call(["git", "checkout", IMS_TOUCAN_COMMIT], cwd=cache_dir)
    if not os.path.exists(marker):
        _patch_ims_toucan(cache_dir)
        open(marker, "w").close()
    if cache_dir not in sys.path:
        sys.path.insert(0, cache_dir)


class ToucanTTSWrapper(nn.Module):
    def __init__(self, model, seq_len):
        super().__init__()
        self.model = model
        self.register_buffer("text_mask", torch.ones(1, 1, seq_len, dtype=torch.bool))

    @torch.no_grad()
    def forward(self, text, text_lengths, utterance_embedding, lang_ids):
        return self.model._forward(
            text_tensors=text,
            text_lengths=text_lengths,
            text_masks=self.text_mask,
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

    def load_model(self, **kwargs):
        _ensure_ims_toucan()
        from huggingface_hub import hf_hub_download
        from Modules.ToucanTTS.InferenceToucanTTS import ToucanTTS

        checkpoint_path = hf_hub_download(
            repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = ToucanTTS(weights=checkpoint["model"], config=checkpoint["config"])
        with torch.no_grad():
            model.store_inverse_all()
        model.eval()
        wrapper = ToucanTTSWrapper(model, seq_len=32)
        wrapper.eval()
        return wrapper

    def load_inputs(self):
        batch = 1
        text_len = 32
        phone_feature_dim = 64

        text = torch.randn(batch, text_len, phone_feature_dim)
        text_lengths = torch.tensor([text_len], dtype=torch.long)
        utterance_embedding = torch.randn(batch, 192)
        lang_ids = torch.tensor([0], dtype=torch.long)

        return text, text_lengths, utterance_embedding, lang_ids
