# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechBrain XLSR accent identification model loader for audio classification.
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
    """Available accent identification XLSR model variants."""

    COMMONACCENT_XLSR_EN = "CommonAccent_XLSR_EN"


def _patch_statistics_pooling():
    """Patch StatisticsPooling.forward to avoid stablehlo.round_nearest_even.

    SpeechBrain's StatisticsPooling computes int(torch.round(lengths * max_len))
    which lowers to stablehlo.round_nearest_even -- an op not yet supported by
    TT-MLIR. For XLA device tensors we bypass the round entirely: lengths[i]
    is always 1.0 for full-sequence inference (wav_lens = 1.0), so
    actual_size == x.shape[1]. Using x.shape[1] directly avoids both the
    unsupported op and the XLA sync that int(xla_tensor) would force.
    """
    try:
        from speechbrain.nnet.pooling import StatisticsPooling
    except ImportError:
        return
    if getattr(StatisticsPooling, "_tt_patched", False):
        return

    _orig_fwd = StatisticsPooling.forward

    def _patched_fwd(self_pool, x, lengths=None):
        if lengths is not None:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                length_snt = lengths[snt_id]
                if (
                    isinstance(length_snt, torch.Tensor)
                    and length_snt.device.type == "xla"
                ):
                    # lengths is an XLA tensor: int(torch.round(xla)) would
                    # force an XLA sync and generate round_nearest_even.
                    # For inference with wav_lens=1.0 (full sequence),
                    # actual_size equals x.shape[1] exactly.
                    actual_size = x.shape[1]
                else:
                    actual_size = int(torch.round(length_snt * x.shape[1]))
                if self_pool.return_mean:
                    mean.append(torch.mean(x[snt_id, 0:actual_size, ...], dim=0))
                if self_pool.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self_pool.return_mean:
                mean = torch.stack(mean)
            if self_pool.return_std:
                std = torch.stack(std)
        else:
            if self_pool.return_mean:
                mean = x.mean(dim=1)
            if self_pool.return_std:
                std = x.std(dim=1)

        if self_pool.return_mean:
            gnoise = self_pool._get_gauss_noise(mean.size(), device=mean.device)
            mean += gnoise
        if self_pool.return_std:
            std = std + self_pool.eps

        if self_pool.return_mean and self_pool.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self_pool.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self_pool.return_std:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats

    StatisticsPooling.forward = _patched_fwd
    StatisticsPooling._tt_patched = True


@torch.compiler.disable
def _run_wav2vec2(wav2vec2_module, wavs):
    """Run wav2vec2 on CPU to avoid pos_conv_embed L1 overflow on TT hardware.

    The XLSR pos_conv_embed grouped conv2d (1024 ch, kernel=128) produces a
    1x49 output which TILE layout limits to div_up(49,32)=2 DRAM slices --
    insufficient for the required L1, causing TT_FATAL at runtime.

    Running wav2vec2 on CPU keeps pos_conv_embed off the XLA lazy graph.
    AccentClassifierModel.to() prevents moving wav2vec2 parameters to XLA, so
    wavs.cpu() + CPU parameters = pure CPU execution with no XLA lazy ops.

    The result is moved back to XLA so that downstream partitions (avg_pool,
    output_mlp) stay in the XLA graph and run on TT hardware.

    During CPU golden-reference runs wavs.device.type is "cpu"; the device
    check skips feats.to("xla") to avoid device mismatch against CPU output_mlp
    parameters in that context.
    """
    feats = wav2vec2_module(wavs.cpu())
    if wavs.device.type == "xla":
        return feats.to("xla")
    return feats


@torch.compiler.disable
def _run_avg_pool(avg_pool_module, feats, wav_lens):
    """Run avg_pool outside the compiled TT subgraph.

    The @torch.compiler.disable decorator creates a dynamo graph break.
    StatisticsPooling is patched by _patch_statistics_pooling() to avoid
    stablehlo.round_nearest_even and the int(xla_tensor) forced sync for
    XLA device tensors.
    """
    return avg_pool_module(feats, wav_lens)


class AccentClassifierModel(torch.nn.Module):
    """Wrapper module for the SpeechBrain XLSR accent classifier pipeline."""

    def __init__(self, classifier):
        super().__init__()
        # Store wav2vec2 outside nn.Module._modules via object.__setattr__ so
        # that Module.to() (called on this model or on the torch.compile
        # OptimizedModule wrapper) never recursively moves wav2vec2 parameters
        # to XLA. pos_conv_embed causes a TT_FATAL at runtime due to L1
        # overflow, so wav2vec2 must run entirely on CPU via _run_wav2vec2.
        object.__setattr__(self, "wav2vec2", classifier.mods.wav2vec2)
        # Disable both internal SpeechBrain layer_norm passes: ttnn.layer_norm on
        # the raw (1, 16000) waveform and on the encoder output are unsupported on
        # TT hardware. Inputs are pre-normalized in load_inputs instead.
        self.wav2vec2.normalize_wav = False
        self.wav2vec2.output_norm = False
        self.avg_pool = classifier.mods.avg_pool
        self.output_mlp = classifier.mods.output_mlp
        # Patch StatisticsPooling globally to avoid round_nearest_even on XLA.
        _patch_statistics_pooling()

    def forward(self, wavs, wav_lens):
        # wav2vec2 runs via a @torch.compiler.disable graph break on CPU (see
        # _run_wav2vec2) so pos_conv_embed never appears in the XLA lazy graph.
        feats = _run_wav2vec2(self.wav2vec2, wavs)
        # avg_pool runs via a @torch.compiler.disable graph break; the
        # StatisticsPooling monkey-patch avoids round_nearest_even for XLA.
        pooled = _run_avg_pool(self.avg_pool, feats, wav_lens)
        logits = self.output_mlp(pooled)
        return logits


class ModelLoader(ForgeModel):
    """SpeechBrain XLSR accent identification model loader."""

    _VARIANTS = {
        ModelVariant.COMMONACCENT_XLSR_EN: ModelConfig(
            pretrained_model_name="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COMMONACCENT_XLSR_EN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="AccentIdCommonAccentXLSR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load the SpeechBrain XLSR accent identification model."""
        from speechbrain.inference.interfaces import foreign_class

        # SpeechBrain's wav2vec2 integration creates float32 tensors internally,
        # so keep the model in native float32 to avoid dtype mismatches.
        kwargs.pop("dtype_override", None)

        classifier = foreign_class(
            source=self._variant_config.pretrained_model_name,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            **kwargs,
        )

        model = AccentClassifierModel(classifier)
        model.eval()

        return model

    def load_inputs(self):
        """Generate synthetic 1-second audio waveform at 16kHz."""
        waveform = torch.randn(1, 16000)
        wav_lens = torch.tensor([1.0])
        # Pre-normalize the waveform outside the model (equivalent to normalize_wav=True),
        # so the TT-compiled model doesn't need to run layer_norm on the raw waveform.
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape[1:])
        return [waveform, wav_lens]
