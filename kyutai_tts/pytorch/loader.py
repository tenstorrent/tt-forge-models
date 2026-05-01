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
        # _undelay_sequence fills delay-masked positions with float('NaN') via
        # in-place assignment.  TT hardware does not preserve bfloat16 NaN; those
        # positions come back as max_bfloat16 (~3.39e38) or Inf.  The wrapper
        # patches _undelay_sequence at load time to use 0.0 fill instead, so
        # out.logits is always finite and we can return it directly.
        return out.logits


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
        import moshi.models.lm as _moshi_lm

        # Patch _undelay_sequence to use 0.0 fill instead of float('NaN').
        # TT hardware (non-IEEE bfloat16) does not preserve NaN; in-place NaN
        # fill produces max_bfloat16 (~3.39e38) or Inf in the compiled graph.
        # Using 0.0 fills delay-masked positions with a finite value that the
        # forward pass can return directly.
        def _undelay_sequence_zeros(delays, tensor, fill_value=float("NaN")):
            B, K, T, *_ = tensor.shape
            assert len(delays) == K
            mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
            outs = []
            if all(d == 0 for d in delays):
                return tensor, mask
            for k, delay in enumerate(delays):
                assert delay >= 0
                line = tensor[:, k].roll(-delay, dims=1)
                if delay > 0:
                    # Functional replacement for in-place NaN fill: zero out the
                    # last `delay` columns without creating non-finite constants.
                    line = torch.cat(
                        [line[:, : T - delay], torch.zeros_like(line[:, :delay])],
                        dim=1,
                    )
                    mask[:, k, -delay:] = 0
                outs.append(line)
            return torch.stack(outs, dim=1), mask

        _moshi_lm._undelay_sequence = _undelay_sequence_zeros

        # Patch _delay_sequence to avoid roll() on int64 code tensors.
        # TT hardware operates in bfloat16; roll() on int64 tensors converts
        # indices to bfloat16 which loses precision for codes > 256 (bfloat16
        # can only represent integers exactly up to 2^8=256; codes can be up
        # to card=2048).  Rounded indices cause wrong embedding lookups in the
        # depformer, producing PCC=0.07-0.39 for all audio codebooks with
        # delay > 0.  Replacing roll+in-place with functional cat+slice avoids
        # any integer arithmetic on TT hardware.
        def _delay_sequence_functional(delays, tensor, padding):
            # padding has shape [B, K, 1] (from _get_initial_token().expand)
            # padding[:, k] is [B, 1]
            B, K, T = tensor.shape
            assert len(delays) == K
            outs = []
            for k, delay in enumerate(delays):
                assert delay >= 0
                if delay == 0:
                    outs.append(tensor[:, k])
                else:
                    pad = padding[:, k].expand(B, delay)  # [B, 1] → [B, delay]
                    outs.append(torch.cat([pad, tensor[:, k, :T - delay]], dim=1))
            return torch.stack(outs, dim=1)

        _moshi_lm._delay_sequence = _delay_sequence_functional

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
