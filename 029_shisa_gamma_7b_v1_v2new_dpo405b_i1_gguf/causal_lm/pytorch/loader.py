# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
029 Shisa Gamma 7B v1 v2new DPO 405B i1 GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Available 029 Shisa Gamma 7B v1 v2new DPO 405B i1 GGUF model variants for causal language modeling."""

    SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF = (
        "029_SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF"
    )


def _prepare_gguf_env():
    """Fix two runtime GGUF issues before any transformers GGUF call.

    1. PACKAGE_DISTRIBUTION_MAPPING fix: transformers caches
       importlib.metadata.packages_distributions() at import time. If gguf is
       installed mid-session the cached map is stale, causing is_gguf_available()
       to return version 'N/A' and packaging.version.InvalidVersion. Refresh it.

    2. model_to_load fix: some model loaders patch load_gguf_checkpoint with a
       signature that omits model_to_load. Newer transformers passes that kwarg,
       raising TypeError. Wrap the live function at call time to accept it.
    """
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.modeling_utils as _modeling_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    import transformers.utils.import_utils as _import_utils

    # Fix 1: refresh stale PACKAGE_DISTRIBUTION_MAPPING
    if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        try:
            _import_utils.PACKAGE_DISTRIBUTION_MAPPING.update(
                importlib.metadata.packages_distributions()
            )
        except Exception:
            pass

    # Fix 2: ensure load_gguf_checkpoint accepts model_to_load
    current = _gguf_utils.load_gguf_checkpoint
    if "model_to_load" not in inspect.signature(current).parameters:
        _captured = current

        def _wrapped(gguf_path, return_tensors=False, model_to_load=None):
            return _captured(gguf_path, return_tensors=return_tensors)

        for _mod in (_gguf_utils, _config_utils, _auto_tokenizer, _modeling_utils):
            _mod.load_gguf_checkpoint = _wrapped


class ModelLoader(ForgeModel):
    """029 Shisa Gamma 7B v1 v2new DPO 405B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/029-shisa-gamma-7b-v1-v2new-dpo405b-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF

    GGUF_FILE = "029-shisa-gamma-7b-v1-v2new-dpo405b.i1-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="029 Shisa Gamma 7B v1 v2new DPO 405B i1 GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _prepare_gguf_env()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

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
        _prepare_gguf_env()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
