# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CodeSage model loader implementation for code embedding generation.
"""

from typing import Optional

import transformers.modeling_utils as _modeling_utils
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

if not hasattr(_modeling_utils, "Conv1D"):
    from transformers.pytorch_utils import Conv1D as _Conv1D

    _modeling_utils.Conv1D = _Conv1D

_orig_init_weights = PreTrainedModel.init_weights


def _patched_init_weights(self, *args, **kwargs):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
            all_submodels=False
        )
    _orig_init_weights(self, *args, **kwargs)


PreTrainedModel.init_weights = _patched_init_weights

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
    """Available CodeSage model variants for code embedding generation."""

    SMALL_V2 = "small-v2"


class ModelLoader(ForgeModel):
    """CodeSage model loader implementation for code embedding generation."""

    _VARIANTS = {
        ModelVariant.SMALL_V2: ModelConfig(
            pretrained_model_name="codesage/codesage-small-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_V2

    sample_code = "def print_hello_world():\n\tprint('Hello World!')"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CodeSage",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True, "add_eos_token": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_ids = self.tokenizer.encode(self.sample_code, return_tensors="pt")

        return {"input_ids": input_ids}
