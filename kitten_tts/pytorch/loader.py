# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KittenTTS model loader implementation for text-to-speech tasks.
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


class KittenTTSWrapper(nn.Module):
    """Wrapper around KittenTTS to expose generate as forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        audio = self.model.generate(text)
        return torch.tensor(audio)


class ModelVariant(StrEnum):
    """Available KittenTTS model variants."""

    KITTEN_TTS_NANO_0_1 = "nano-0.1"
    KITTEN_TTS_MINI_0_1 = "mini-0.1"


class ModelLoader(ForgeModel):
    """KittenTTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KITTEN_TTS_NANO_0_1: ModelConfig(
            pretrained_model_name="KittenML/kitten-tts-nano-0.1",
        ),
        ModelVariant.KITTEN_TTS_MINI_0_1: ModelConfig(
            pretrained_model_name="KittenML/kitten-tts-mini-0.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KITTEN_TTS_NANO_0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tts_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KittenTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _ensure_espeakng_data():
        """Ensure espeak-ng data is accessible at the path compiled into espeakng_loader.

        The espeakng_loader wheel bundles espeak-ng compiled against a
        GitHub Actions runner path. On non-CI machines that path doesn't exist,
        so we create a symlink from the compiled-in path to the bundled data.
        """
        import os
        import subprocess

        try:
            import espeakng_loader

            lib_path = espeakng_loader.get_library_path()
            data_path = espeakng_loader.get_data_path()

            # Find the hardcoded prefix path compiled into the binary
            result = subprocess.run(
                ["strings", lib_path], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                # The lib directory compiled into the binary (e.g.
                # /home/runner/.../espeak-ng/_dynamic/lib)
                if line.startswith("/") and line.endswith("/lib") and "espeak" in line:
                    hardcoded_data = os.path.join(
                        os.path.dirname(line), "share", "espeak-ng-data"
                    )
                    if not os.path.exists(hardcoded_data):
                        os.makedirs(os.path.dirname(hardcoded_data), exist_ok=True)
                        os.symlink(data_path, hardcoded_data)
                    break
        except Exception:
            pass  # best-effort; let espeak_Initialize raise if still broken

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_espeakng_data()
        from kittentts import KittenTTS

        self.tts_model = KittenTTS(self._variant_config.pretrained_model_name)
        model = KittenTTSWrapper(self.tts_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        text = "Hello, this is a test of text to speech."
        return (text,)
