# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kyutai TTS 1.6B streaming text-to-speech model loader implementation.
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


class _LMForwardWrapper(torch.nn.Module):
    """Wraps LMModel with pre-computed dummy condition tensors as buffers.

    TTSModel.lm requires condition_tensors (speaker_wavs, cfg, control) for
    every forward call. This wrapper pre-computes dummy conditions at load
    time and stores them as registered buffers so they move with the model
    to the TT device.
    """

    def __init__(self, lm, condition_tensors):
        super().__init__()
        self.lm = lm
        for name, (cond, mask) in condition_tensors.items():
            self.register_buffer(f"cond_{name}", cond.detach())
            self.register_buffer(f"mask_{name}", mask.detach())
        self._condition_names = list(condition_tensors.keys())

    def forward(self, codes):
        cond_tensors = {
            name: (getattr(self, f"cond_{name}"), getattr(self, f"mask_{name}"))
            for name in self._condition_names
        }
        out = self.lm(codes, cond_tensors)
        # _undelay_sequence fills delay-masked positions with float('NaN').
        # TT hardware uses non-IEEE bfloat16 that does not preserve NaN; those
        # positions come back as max_bfloat16 (~3.39e38) instead.  Use the
        # model's own mask (already computed via bool operations, which are
        # unaffected) to explicitly zero out invalid positions on both CPU and TT.
        # out.mask: [B, K, T], True = valid position
        return torch.where(out.mask.unsqueeze(-1), out.logits, torch.zeros_like(out.logits))


class ModelVariant(StrEnum):
    """Available Kyutai TTS model variants."""

    TTS_1_6B_EN_FR = "TTS 1.6B en_fr"


class ModelLoader(ForgeModel):
    """Kyutai TTS 1.6B streaming text-to-speech model loader implementation."""

    _VARIANTS = {
        ModelVariant.TTS_1_6B_EN_FR: ModelConfig(
            pretrained_model_name="kyutai/tts-1.6b-en_fr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TTS_1_6B_EN_FR

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KyutaiTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kyutai TTS LM backbone wrapped with dummy conditions."""
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel, ConditionAttributes, TensorCondition

        pretrained_model_name = self._variant_config.pretrained_model_name

        checkpoint_info = CheckpointInfo.from_hf_repo(pretrained_model_name)
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=32, temp=0.6, device=torch.device("cpu"), dtype=dtype
        )
        lm = tts_model.lm
        lm.eval()

        self._num_codebooks = lm.num_codebooks
        self._card = lm.card

        # Pre-compute dummy condition tensors. speaker_wavs input dim is 512
        # (TensorConditioner.output_proj: 512→2048). cfg/control are LUT text conditioners.
        attrs = ConditionAttributes(
            text={"cfg": "1.0", "control": "ok"},
            tensor={
                "speaker_wavs": TensorCondition(
                    torch.zeros(1, 1, 512, dtype=torch.float32),
                    torch.ones(1, 1, dtype=torch.bool),
                )
            },
        )
        with torch.no_grad():
            condition_tensors = lm.condition_provider.prepare_and_provide([attrs])

        return _LMForwardWrapper(lm, condition_tensors)

    def load_inputs(self, dtype_override=None):
        """Load synthetic discrete code inputs for the Kyutai TTS model.

        The model expects discrete codes of shape [B, K, T] where
        K is the number of codebooks and T is time steps.
        """
        codes = torch.randint(0, self._card, (1, self._num_codebooks, 10))
        return codes
