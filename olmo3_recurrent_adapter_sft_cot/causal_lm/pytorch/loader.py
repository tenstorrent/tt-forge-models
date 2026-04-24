# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Olmo3 Recurrent Adapter SFT CoT causal LM model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers.integrations.accelerate as _acc_module
from transformers.modeling_utils import get_torch_context_manager_or_global_device
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
    """Available Olmo3 Recurrent Adapter SFT CoT model variants for causal language modeling."""

    Olmo3_Recurrent_Adapter_SFT_CoT = "recurrent_adapter_sft_cot"


class ModelLoader(ForgeModel):
    """Olmo3 Recurrent Adapter SFT CoT model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.Olmo3_Recurrent_Adapter_SFT_CoT: LLMModelConfig(
            pretrained_model_name="hanseungwook/olmo3-recurrent-adapter-sft-cot",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Olmo3_Recurrent_Adapter_SFT_CoT

    # The recurrent adapter reuses the OLMo-3-1025-7B base tokenizer.
    tokenizer_model_name = "allenai/Olmo-3-1025-7B"

    sample_text = "What is 25 * 37?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="olmo3_recurrent_adapter_sft_cot",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        # The model's __init__ calls from_pretrained internally for the base model.
        # Newer transformers wraps __init__ in a torch.device("meta") context, which
        # causes the nested from_pretrained to raise. Patch check_and_set_device_map
        # to redirect to CPU instead of raising when inside a meta device context.
        _orig_check = _acc_module.check_and_set_device_map

        def _allow_nested_from_pretrained(device_map):
            if (
                device_map is None
                and get_torch_context_manager_or_global_device() == torch.device("meta")
            ):
                return {"": torch.device("cpu")}
            return _orig_check(device_map)

        _acc_module.check_and_set_device_map = _allow_nested_from_pretrained
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            _acc_module.check_and_set_device_map = _orig_check
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.config
