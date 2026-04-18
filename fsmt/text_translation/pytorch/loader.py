# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FSMT model loader implementation for text translation.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available FSMT model variants for text translation."""

    WMT19_RU_EN = "Wmt19_Ru_En"


class ModelLoader(ForgeModel):
    """FSMT model loader implementation for text translation."""

    _VARIANTS = {
        ModelVariant.WMT19_RU_EN: LLMModelConfig(
            pretrained_model_name="facebook/wmt19-ru-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WMT19_RU_EN

    _SAMPLE_TEXTS = {
        ModelVariant.WMT19_RU_EN: "Машинное обучение - это здорово, не так ли?",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="FSMT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import FSMTTokenizer

        self._tokenizer = FSMTTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._tokenizer

    @staticmethod
    def _patch_fsmt_source():
        """Patch modeling_fsmt.py to create causal mask on the correct device."""
        import transformers.models.fsmt.modeling_fsmt as fsmt_module
        import inspect
        import importlib
        import pathlib

        source_file = inspect.getfile(fsmt_module)
        with open(source_file, "r") as f:
            content = f.read()

        old = (
            "torch.zeros(tgt_len, tgt_len, dtype=causal_mask_dtype)), 1).to(\n"
            "        device=decoder_input_ids.device\n"
            "    )"
        )
        new = "torch.zeros(tgt_len, tgt_len, dtype=causal_mask_dtype, device=decoder_input_ids.device)), 1)"

        if old in content:
            content = content.replace(old, new)
            with open(source_file, "w") as f:
                f.write(content)
            for pyc in pathlib.Path(source_file).parent.glob(
                "__pycache__/modeling_fsmt*"
            ):
                pyc.unlink(missing_ok=True)
            importlib.reload(fsmt_module)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FSMT model instance for this instance's variant."""
        self._patch_fsmt_source()

        from transformers import FSMTForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FSMTForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the FSMT model."""
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        sample_text = self._SAMPLE_TEXTS.get(self._variant)
        inputs = self._tokenizer(
            sample_text,
            return_tensors="pt",
        )

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
