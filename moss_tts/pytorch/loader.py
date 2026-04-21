# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOSS-TTS-Realtime model loader implementation for text-to-speech tasks.
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


class MossTTSRealtimeLanguageWrapper(nn.Module):
    """Wrapper around the MOSS-TTS-Realtime Qwen3 language backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces hidden states from the language model.
    """

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return outputs.last_hidden_state


class ModelVariant(StrEnum):
    """Available MOSS-TTS-Realtime model variants."""

    MOSS_TTS_REALTIME_1_7B = "1.7B"


class ModelLoader(ForgeModel):
    """MOSS-TTS-Realtime model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MOSS_TTS_REALTIME_1_7B: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-TTS-Realtime",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOSS_TTS_REALTIME_1_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MOSS-TTS-Realtime",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _register_custom_classes():
        import importlib.util
        import os
        import sys
        import types

        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel

        if "_moss_tts_hub" in sys.modules:
            return

        repo = "OpenMOSS-Team/MOSS-TTS-Realtime"
        pkg_name = "_moss_tts_hub"

        init_path = hf_hub_download(repo, "__init__.py")
        snapshot_dir = os.path.dirname(init_path)

        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [snapshot_dir]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

        module_names = [
            "configuration_mossttsrealtime",
            "modeling_mossttsrealtime_local",
            "modeling_mossttsrealtime",
        ]
        for mod_name in module_names:
            fqn = f"{pkg_name}.{mod_name}"
            fpath = os.path.join(snapshot_dir, f"{mod_name}.py")
            hf_hub_download(repo, f"{mod_name}.py")
            spec = importlib.util.spec_from_file_location(fqn, fpath)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg_name
            sys.modules[fqn] = mod
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)

        cfg_mod = sys.modules["configuration_mossttsrealtime"]
        mdl_mod = sys.modules["modeling_mossttsrealtime"]

        AutoConfig.register("moss_tts_realtime", cfg_mod.MossTTSRealtimeConfig)
        AutoModel.register(cfg_mod.MossTTSRealtimeConfig, mdl_mod.MossTTSRealtime)

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        self._register_custom_classes()

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_override or torch.float32,
        )
        model = MossTTSRealtimeLanguageWrapper(full_model.language_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Qwen3 language backbone hidden_size=2048, use a short sequence
        inputs_embeds = torch.randn(1, 32, 2048, dtype=dtype)
        return (inputs_embeds,)
