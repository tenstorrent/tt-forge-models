# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CodeSage Large v2 model loader for code embedding generation.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)


class ModelVariant(StrEnum):
    """Available model variants for CodeSage Large v2."""

    CODESAGE_LARGE_V2 = "codesage/codesage-large-v2"


class ModelLoader(ForgeModel):
    """CodeSage Large v2 model loader for code embedding generation."""

    _VARIANTS = {
        ModelVariant.CODESAGE_LARGE_V2: LLMModelConfig(
            pretrained_model_name="codesage/codesage-large-v2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CODESAGE_LARGE_V2

    sample_code = "def print_hello_world():\tprint('Hello World!')"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CodeSage-Large-v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
                add_eos_token=True,
            )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        import transformers
        import transformers.modeling_utils

        # transformers 5.x removed Conv1D from modeling_utils; the codesage
        # custom code still imports it from there, so restore it before loading.
        if not hasattr(transformers.modeling_utils, "Conv1D"):
            from transformers.pytorch_utils import Conv1D

            transformers.modeling_utils.Conv1D = Conv1D

        # The codesage custom code calls self.init_weights() from __init__ (old
        # pattern). In transformers 5.x, post_init() must be called first to
        # initialize all_tied_weights_keys before tie_weights() is invoked.
        from transformers import PreTrainedModel

        _orig_init_weights = PreTrainedModel.init_weights

        def _patched_init_weights(self_model):
            if not hasattr(self_model, "all_tied_weights_keys"):
                self_model.all_tied_weights_keys = (
                    self_model.get_expanded_tied_weights_keys(all_submodels=False)
                )
            _orig_init_weights(self_model)

        PreTrainedModel.init_weights = _patched_init_weights

        from transformers import AutoModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        try:
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        finally:
            PreTrainedModel.init_weights = _orig_init_weights

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_code,
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
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

        # CodeSage produces a pooled embedding at the final (eos) token.
        embedding = last_hidden_state[:, -1, :]
        return embedding
