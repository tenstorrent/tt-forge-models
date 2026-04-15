# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi-Audio model loader implementation for audio understanding and generation tasks.
"""
import importlib
import sys
import types

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


class ModelVariant(StrEnum):
    """Available Kimi-Audio model variants."""

    KIMI_AUDIO_7B_INSTRUCT = "7B_Instruct"


class ModelLoader(ForgeModel):
    """Kimi-Audio model loader implementation for audio understanding and generation tasks."""

    _VARIANTS = {
        ModelVariant.KIMI_AUDIO_7B_INSTRUCT: ModelConfig(
            pretrained_model_name="moonshotai/Kimi-Audio-7B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_AUDIO_7B_INSTRUCT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KimiAudio",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _mock_flash_attn():
        """Install mock flash_attn modules so the HF dynamic code can be imported.

        The upstream ``modeling_moonshot_kimia.py`` unconditionally imports
        ``flash_attn`` at module scope and raises if it is missing.  We only
        need the model graph for compilation, so placeholder stubs are enough.
        """
        if "flash_attn" not in sys.modules:
            flash_attn = types.ModuleType("flash_attn")
            flash_attn.__version__ = "2.7.0"
            flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
            flash_attn.flash_attn_func = None
            flash_attn.flash_attn_varlen_func = None
            sys.modules["flash_attn"] = flash_attn

        if "flash_attn.bert_padding" not in sys.modules:
            bert_padding = types.ModuleType("flash_attn.bert_padding")
            bert_padding.__spec__ = importlib.machinery.ModuleSpec(
                "flash_attn.bert_padding", None
            )
            bert_padding.index_first_axis = None
            bert_padding.pad_input = None
            bert_padding.unpad_input = None
            sys.modules["flash_attn.bert_padding"] = bert_padding

        import transformers.utils
        import transformers.utils.import_utils

        _true = lambda *a, **kw: True
        for mod in (transformers.utils, transformers.utils.import_utils):
            mod.is_flash_attn_2_available = _true
            mod.is_flash_attn_available = _true
            if hasattr(mod, "is_flash_attn_greater_or_equal_2_10"):
                mod.is_flash_attn_greater_or_equal_2_10 = _true
            if hasattr(mod, "is_flash_attn_greater_or_equal"):
                mod.is_flash_attn_greater_or_equal = _true

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kimi-Audio model instance."""
        from transformers import AutoConfig, AutoModelForCausalLM

        self._mock_flash_attn()

        pretrained_model_name = self._variant_config.pretrained_model_name

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        # transformers 5.x merged rope_theta/rope_scaling into rope_parameters.
        # The upstream dynamic module still reads config.rope_theta directly.
        if not hasattr(config, "rope_theta"):
            rope_params = getattr(config, "rope_parameters", {}) or {}
            config.rope_theta = rope_params.get("rope_theta", 10000.0)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype_override if dtype_override is not None else torch.float32,
            **kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Kimi-Audio model."""
        from transformers import AutoTokenizer

        # The upstream custom TikTokenTokenizer is incompatible with
        # transformers 5.x.  Use the base Qwen2.5 tokenizer instead since
        # KimiAudio is built on Qwen2.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

        prompt = "Please transcribe the following audio."
        inputs = tokenizer(prompt, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
