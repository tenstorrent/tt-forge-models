# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
F5-TTS Russian model loader implementation for text-to-speech tasks.
"""
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


class F5TTSDiTWrapper(nn.Module):
    """Wrapper around the F5-TTS DiT transformer backbone.

    Exposes the DiT forward pass that takes noised mel, conditioning mel,
    text tokens, and a diffusion timestep to predict the flow vector.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, x, cond, text, time):
        return self.transformer(x=x, cond=cond, text=text, time=time)


class ModelVariant(StrEnum):
    """Available F5-TTS Russian model variants."""

    V1_BASE = "v1_base"


class ModelLoader(ForgeModel):
    """F5-TTS Russian model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.V1_BASE: ModelConfig(
            pretrained_model_name="Misha24-10/F5-TTS_RUSSIAN",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._cfm_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="F5-TTS-Russian",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from f5_tts.model.backbones.dit import DiT
        from f5_tts.model import CFM
        from f5_tts.model.utils import get_tokenizer
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="F5TTS_v1_Base/model_240000_inference.safetensors",
        )
        vocab_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="F5TTS_v1_Base/vocab.txt",
        )

        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )

        vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
        cfm_model = CFM(
            transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
            mel_spec_kwargs=dict(
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mel_channels=100,
                target_sample_rate=24000,
                mel_spec_type="vocos",
            ),
            odeint_kwargs=dict(method="euler"),
            vocab_char_map=vocab_char_map,
        )

        checkpoint = load_file(ckpt_path, device="cpu")
        checkpoint = {"ema_model_state_dict": checkpoint}
        state_dict = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        for key in [
            "mel_spec.mel_stft.mel_scale.fb",
            "mel_spec.mel_stft.spectrogram.window",
        ]:
            state_dict.pop(key, None)
        cfm_model.load_state_dict(state_dict)

        self._cfm_model = cfm_model

        model = F5TTSDiTWrapper(cfm_model.transformer)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the DiT transformer backbone.

        Returns:
            tuple: (x, cond, text, time) tensors for the DiT forward pass.
                - x: Noised mel spectrogram [batch, seq_len, mel_dim].
                - cond: Conditioning mel spectrogram [batch, seq_len, mel_dim].
                - text: Token indices [batch, seq_len].
                - time: Diffusion timestep [batch].
        """
        dtype = dtype_override or torch.float32
        batch_size = 1
        seq_len = 64
        mel_dim = 100

        x = torch.randn(batch_size, seq_len, mel_dim, dtype=dtype)
        cond = torch.randn(batch_size, seq_len, mel_dim, dtype=dtype)
        text = torch.randint(0, 256, (batch_size, seq_len))
        time = torch.tensor([0.5], dtype=dtype)
        return x, cond, text, time
