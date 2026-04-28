# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Braindecode BENDR model loader implementation for EEG classification.

BENDR (BErt-inspired Neural Data Representations) is a pretrained
transformer-based model for EEG classification tasks using self-supervised
learning on masked sequence reconstruction.
"""

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
    """Available Braindecode BENDR model variants."""

    BENDR_PRETRAINED = "bendr_pretrained"


class ModelLoader(ForgeModel):
    """Braindecode BENDR model loader implementation for EEG classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BENDR_PRETRAINED: ModelConfig(
            pretrained_model_name="braindecode/braindecode-bendr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BENDR_PRETRAINED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BENDR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from braindecode.models import BENDR
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model = BENDR(n_chans=20, n_outputs=2)

        checkpoint_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.safetensors",
        )
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)
            # braindecode's Contextualizer creates the start_token via torch.full()
            # without a dtype argument, so it defaults to float32.  The subsequent
            # torch.cat promotes the sequence tensor to float32, which then mismatches
            # the bfloat16 transformer weights.  Fix: cast each transformer layer's
            # input back to the requested dtype before the linear projection.
            _dt = dtype_override
            for layer in model.contextualizer.transformer_layers:
                layer.register_forward_pre_hook(
                    lambda _, inp: (inp[0].to(_dt),) + inp[1:]
                )

        return model

    def load_inputs(self, dtype_override=None):
        import torch

        # BENDR expects raw EEG data: (batch, channels, time_samples)
        # 20 EEG channels, 600 time samples (~1 second at typical sampling rates)
        n_chans = 20
        n_samples = 600
        eeg_data = torch.randn(1, n_chans, n_samples)

        if dtype_override is not None:
            eeg_data = eeg_data.to(dtype_override)

        return [eeg_data]
