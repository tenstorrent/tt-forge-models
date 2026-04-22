# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek R1 NextN model loader implementation for causal language modeling.

The NextN layer is a speculative decoding draft module exported from
DeepSeek-R1, intended for use with the EAGLE algorithm in SGLang.
"""
import json
import os
from typing import Optional

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
    """Available DeepSeek R1 NextN model variants."""

    DEEPSEEK_R1_NEXTN = "DeepSeek-R1-NextN"


class ModelLoader(ForgeModel):
    """DeepSeek R1 NextN model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_R1_NEXTN: LLMModelConfig(
            pretrained_model_name="lmsys/DeepSeek-R1-NextN",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_R1_NEXTN

    sample_text = "The future of artificial intelligence is"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-R1-NextN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _patch_index_file(self, pretrained_model_name):
        """Add missing 'metadata' key to model.safetensors.index.json.

        lmsys/DeepSeek-R1-NextN omits this key, which transformers requires.
        """
        from huggingface_hub import hf_hub_download

        index_path = hf_hub_download(
            pretrained_model_name, "model.safetensors.index.json"
        )
        real_path = os.path.realpath(index_path)

        with open(real_path) as f:
            index = json.load(f)

        if "metadata" not in index:
            index["metadata"] = {"total_size": 0}
            with open(real_path, "w") as f:
                json.dump(index, f)

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # lmsys/DeepSeek-R1-NextN has no modeling_deepseek.py (trust_remote_code won't
        # work) and its model.safetensors.index.json is missing the 'metadata' key
        # required by transformers. Patch the cached index before loading.
        self._patch_index_file(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
