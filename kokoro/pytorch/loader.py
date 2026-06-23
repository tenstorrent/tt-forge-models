# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.

Kokoro is a StyleTTS2-derived TTS model (PL-BERT prosody encoder + duration
predictor + iSTFTNet decoder). The architecture is vendored under ``src/`` from
the ``kokoro`` package (Apache-2.0). The loader builds the model from
``config.json`` with random-initialized weights (no multi-GB checkpoint
download) and exposes a clean tensors-in / tensor-out forward suitable for
compile bringup and PCC comparison.
"""
import json
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

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
from .src.model import KModel


class ModelVariant(StrEnum):
    """Available Kokoro model variants."""

    KOKORO_82M = "hexgrad/Kokoro-82M"


class _KokoroWrapper(torch.nn.Module):
    """Tensors-only wrapper around KModel for compile / PCC.

    KModel.forward takes a phoneme *string*; this wrapper drives the underlying
    ``forward_with_tokens`` so the model is tensor-in (input_ids, ref_s) and
    returns the audio waveform tensor.
    """

    def __init__(self, kmodel: KModel, speed: float = 1.0):
        super().__init__()
        self.kmodel = kmodel
        self.speed = speed

    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor):
        audio, _pred_dur = self.kmodel.forward_with_tokens(input_ids, ref_s, self.speed)
        return audio


class ModelLoader(ForgeModel):
    """Kokoro-82M model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KOKORO_82M: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KOKORO_82M

    # A short phoneme sequence (token ids into the model vocab). Length kept
    # small and well under the 512-token context window.
    #
    # NOTE: reduced from 48 -> 8 for compile tractability. The duration
    # predictor expands each token into many audio frames, so 48 tokens yields
    # a ~780K-sample 1-D waveform; the conv-based inverse-STFT graph over that
    # length does not finish tt-mlir compilation within 40 min (graph-size
    # explosion). 8 tokens (~150K samples) compiles and runs end-to-end on n150.
    DEFAULT_NUM_TOKENS = 8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="kokoro",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_config(self):
        if self._config is None:
            config_path = hf_hub_download(
                repo_id=self._variant_config.pretrained_model_name,
                filename="config.json",
            )
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        return self._config

    def load_model(self, dtype_override=None):
        """Build Kokoro from config and load the trained checkpoint.

        ``load_weights=True`` downloads + loads ``kokoro-v1_0.pth``. Trained
        weights are required for meaningful PCC: with random init the iSTFTNet
        vocoder's AdaIN ``InstanceNorm`` sees a near-constant activation
        (per-channel variance ~5e-8) whose bf16 variance catastrophically
        cancels on device -> ``rsqrt`` -> inf, exploding the output to FLT_MAX
        and collapsing PCC. Trained ``noise_convs`` carry real variance
        (~3e-3), so the normalization (and the downstream ``torch.exp``) is
        well conditioned.

        ``disable_complex=True`` selects the conv-based CustomSTFT path (no
        complex tensor ops), which is friendlier to the TT compiler.

        Kokoro is pinned to fp32. The iSTFTNet vocoder (sine-source phase
        accumulation + conv-based STFT/iSTFT over a long 1-D waveform) is highly
        numerically sensitive: running it in bf16 collapses waveform-level PCC to
        ~0.18 even on CPU with identical weights (fp32 vs bf16), so any ``bf16``
        ``dtype_override`` from the runner is intentionally ignored here.
        """
        config = self._load_config()
        kmodel = KModel(
            repo_id=self._variant_config.pretrained_model_name,
            config=config,
            disable_complex=True,
            load_weights=True,
        )
        model = _KokoroWrapper(kmodel).eval()
        # Pin fp32 (see docstring); only honor a non-bf16 explicit override.
        if dtype_override is not None and dtype_override != torch.bfloat16:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Synthesize matched inputs for the wrapped forward.

        - ``input_ids``: phoneme token ids in [1, n_token) wrapped by BOS/EOS 0s.
        - ``ref_s``: speaker/style reference vector of width ``2 * style_dim``
          (first half = decoder style, second half = predictor style).
        """
        config = self._load_config()
        n_token = config["n_token"]
        style_dim = config["style_dim"]

        torch.manual_seed(0)
        n = self.DEFAULT_NUM_TOKENS
        inner = torch.randint(low=1, high=n_token, size=(n,), dtype=torch.long)
        input_ids = torch.cat(
            [torch.zeros(1, dtype=torch.long), inner, torch.zeros(1, dtype=torch.long)]
        ).unsqueeze(0)

        # Keep inputs fp32 to match the fp32-pinned model (see load_model).
        ref_s = torch.randn(1, 2 * style_dim)
        if dtype_override is not None and dtype_override != torch.bfloat16:
            ref_s = ref_s.to(dtype_override)

        return {"input_ids": input_ids, "ref_s": ref_s}
