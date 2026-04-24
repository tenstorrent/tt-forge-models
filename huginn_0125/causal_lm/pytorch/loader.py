# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huginn-0125 model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available Huginn-0125 model variants."""

    HUGINN_0125 = "0125"


class ModelLoader(ForgeModel):
    """Huginn-0125 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUGINN_0125: LLMModelConfig(
            pretrained_model_name="tomg-group-umd/huginn-0125",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUGINN_0125

    sample_text = "The capital of Westphalia is"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Huginn-0125",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.n_layers = self.num_layers
            model_kwargs["config"] = config

        # huginn uses trust_remote_code with _tied_weights_keys as a list (old transformers format),
        # but transformers>=4.46 expects a dict. Patch to handle the list case gracefully.
        from transformers import PreTrainedModel as _PTM

        _orig_get_expanded = _PTM.get_expanded_tied_weights_keys

        def _patched_get_expanded(self_inner, all_submodels=False):
            if isinstance(self_inner._tied_weights_keys, list):
                self_inner._tied_weights_keys = None
            return _orig_get_expanded(self_inner, all_submodels=all_submodels)

        _PTM.get_expanded_tied_weights_keys = _patched_get_expanded
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, trust_remote_code=True, **model_kwargs
            )
        finally:
            _PTM.get_expanded_tied_weights_keys = _orig_get_expanded
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text, return_tensors="pt", return_token_type_ids=False
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
