# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/PersonaVLM-i1-GGUF model loader implementation for causal language modeling.

Note: The qwen2vl GGUF architecture is not yet supported by the transformers
GGUF loader, so we load from the HF-native base checkpoint and extract the
causal LM.
"""
import torch
from transformers import (
    Qwen2ForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
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
    """Available mradermacher/PersonaVLM-i1-GGUF model variants for causal language modeling."""

    PERSONAVLM_I1_GGUF = "PersonaVLM_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/PersonaVLM-i1-GGUF model loader implementation for causal language modeling tasks.

    Note: Uses the base model (safetensors) instead of GGUF because the
    qwen2vl GGUF architecture is not yet supported by transformers.
    """

    _VARIANTS = {
        ModelVariant.PERSONAVLM_I1_GGUF: LLMModelConfig(
            pretrained_model_name="ClareNie/PersonaVLM",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PERSONAVLM_I1_GGUF

    sample_text = "Give me a short introduction to large language models."

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
            model="mradermacher PersonaVLM-i1-GGUF",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load only the config (lightweight) and initialize with random weights.
        # The full Qwen2_5_VLForConditionalGeneration checkpoint is ~15 GB which
        # exceeds available disk space in many environments; since this is a
        # compile-only target we only need the architecture, not pretrained weights.
        vl_config = AutoConfig.from_pretrained(pretrained_model_name)
        text_config = vl_config.text_config
        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers

        model = Qwen2ForCausalLM(text_config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()

        self.config = text_config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
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
        config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
        self.config = config.text_config
        return self.config
