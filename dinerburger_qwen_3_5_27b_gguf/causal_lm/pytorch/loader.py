# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dinerburger Qwen3.5-27B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available dinerburger Qwen3.5-27B GGUF model variants for causal language modeling."""

    QWEN_3_5_27B_IQ4_NL = "27B_IQ4_NL"
    QWEN_3_5_27B_IQ4_XS = "27B_IQ4_XS"
    QWEN_3_5_27B_Q5_K = "27B_Q5_K"
    QWEN_3_5_27B_Q8_0_XXL = "27B_Q8_0_XXL"


class ModelLoader(ForgeModel):
    """dinerburger Qwen3.5-27B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B_IQ4_NL: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_IQ4_XS: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_Q5_K: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_Q8_0_XXL: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B_IQ4_NL

    _GGUF_FILES = {
        ModelVariant.QWEN_3_5_27B_IQ4_NL: "Qwen3.5-27B.IQ4_NL.gguf",
        ModelVariant.QWEN_3_5_27B_IQ4_XS: "Qwen3.5-27B.IQ4_XS.gguf",
        ModelVariant.QWEN_3_5_27B_Q5_K: "Qwen3.5-27B.Q5_K.gguf",
        ModelVariant.QWEN_3_5_27B_Q8_0_XXL: "Qwen3.5-27B.Q8_0_XXL.gguf",
    }

    @property
    def GGUF_FILE(self):
        return self._GGUF_FILES[self._variant]

    sample_text = "Give me a short introduction to large language model."

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
            model="dinerburger Qwen3.5-27B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager."""
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
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
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
        self._fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
