# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CodeSage-Small-v2 model loader implementation for code embedding generation.
"""
import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available CodeSage-Small-v2 model variants for code embedding generation."""

    CODESAGE_SMALL_V2 = "codesage-small-v2"


class ModelLoader(ForgeModel):
    """CodeSage-Small-v2 model loader implementation for code embedding generation."""

    _VARIANTS = {
        ModelVariant.CODESAGE_SMALL_V2: ModelConfig(
            pretrained_model_name="codesage/codesage-small-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CODESAGE_SMALL_V2

    sample_code = "def print_hello_world():\n    print('Hello World!')"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CodeSage-Small-v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            add_eos_token=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_code,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs

        # CodeSage returns a 1024-dim embedding from the first position.
        return last_hidden_state[:, 0, :]
