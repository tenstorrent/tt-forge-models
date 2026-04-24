# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Acoustic Bench SALM model loader implementation for speech recognition (ASR) using PyTorch.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Acoustic Bench PyTorch speech recognition model variants."""

    ACOUSTIC_BENCH = "Acoustic_Bench"


class ModelLoader(ForgeModel):
    """Acoustic Bench SALM model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.ACOUSTIC_BENCH: ModelConfig(
            pretrained_model_name="KarthikSivaramaKrishnan/acoustic-bench",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ACOUSTIC_BENCH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Acoustic_Bench",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from nemo.collections.speechlm2.models import SALM

        cfg = {
            "pretrained_asr": "nvidia/canary-1b-flash",
            "pretrained_llm": "TinyLlama/TinyLlama_v1.1",
            "pretrained_weights": False,
            "prompt_format": "llama2",
            "audio_locator_tag": "<|audioplaceholder|>",
            "perception": {
                "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                "output_dim": 2048,
                "encoder": {
                    "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                    "att_context_size": [-1, -1],
                    "causal_downsampling": False,
                    "conv_context_size": None,
                    "conv_kernel_size": 9,
                    "conv_norm_type": "batch_norm",
                    "d_model": 1024,
                    "dropout": 0.1,
                    "dropout_att": 0.1,
                    "dropout_emb": 0.0,
                    "dropout_pre_encoder": 0.1,
                    "feat_in": 128,
                    "feat_out": -1,
                    "ff_expansion_factor": 4,
                    "n_heads": 8,
                    "n_layers": 2,
                    "pos_emb_max_len": 5000,
                    "self_attention_model": "rel_pos",
                    "subsampling": "dw_striding",
                    "subsampling_conv_channels": 256,
                    "subsampling_factor": 8,
                },
                "modality_adapter": {
                    "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                    "d_model": 1024,
                },
                "preprocessor": {
                    "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                    "dither": 1e-05,
                    "features": 128,
                    "frame_splicing": 1,
                    "log": True,
                    "n_fft": 512,
                    "normalize": "per_feature",
                    "pad_to": 0,
                    "pad_value": 0.0,
                    "sample_rate": 16000,
                    "window": "hann",
                    "window_size": 0.025,
                    "window_stride": 0.01,
                },
            },
            "optimizer": {"_target_": "torch.optim.AdamW"},
        }
        model = SALM(cfg)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        batch_size = 1
        seq_len = 128
        hidden_size = 2048  # TinyLlama hidden size

        input_embeds = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        if dtype_override is not None:
            input_embeds = input_embeds.to(dtype_override)

        return {
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
        }
