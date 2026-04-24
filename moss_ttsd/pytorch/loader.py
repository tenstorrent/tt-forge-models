# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOSS-TTSD model loader implementation for text-to-speech dialogue synthesis.
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


class MossTTSDLanguageWrapper(nn.Module):
    """Wrapper around the MOSS-TTSD Qwen3 language backbone.

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
    """Available MOSS-TTSD model variants."""

    MOSS_TTSD_V0_5 = "v0.5"


class ModelLoader(ForgeModel):
    """MOSS-TTSD model loader implementation for text-to-speech dialogue synthesis."""

    _VARIANTS = {
        ModelVariant.MOSS_TTSD_V0_5: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-TTSD-v0.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOSS_TTSD_V0_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MOSS-TTSD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        # transformers>=5 no longer sets pad_token_id by default on PretrainedConfig;
        # derive it from pad_token list in config.json (first element is the text pad token).
        if not hasattr(config, "pad_token_id"):
            if hasattr(config, "pad_token") and isinstance(
                config.pad_token, (list, tuple)
            ):
                config.pad_token_id = config.pad_token[0]
            else:
                config.pad_token_id = getattr(config, "bos_token_id", None)

        # AutoModel maps to MossTTSDForCausalLM which has a tie_weights() incompatible
        # with transformers>=5 (missing recompute_mapping kwarg). Load MossTTSDModel
        # directly instead — it has .language_model and uses the base tie_weights().
        MossTTSDModelClass = get_class_from_dynamic_module(
            "modeling_moss_ttsd.MossTTSDModel",
            self._variant_config.pretrained_model_name,
        )
        full_model = MossTTSDModelClass.from_pretrained(
            self._variant_config.pretrained_model_name,
            config=config,
            torch_dtype=dtype_override or torch.float32,
        )
        model = MossTTSDLanguageWrapper(full_model.language_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Qwen3-1.7B-Base language backbone hidden_size=2048, use a short sequence
        inputs_embeds = torch.randn(1, 32, 2048, dtype=dtype)
        return (inputs_embeds,)
