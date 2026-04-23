# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
lmstudio-community/gpt-oss-safeguard-120b-MLX-MXFP4 model loader for causal language modeling.
"""
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
    """Available lmstudio-community GPT-OSS Safeguard 120B MLX MXFP4 model variants."""

    GPT_OSS_SAFEGUARD_120B_MLX_MXFP4 = "GPT_OSS_SAFEGUARD_120B_MLX_MXFP4"


class ModelLoader(ForgeModel):
    """lmstudio-community GPT-OSS Safeguard 120B MLX MXFP4 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GPT_OSS_SAFEGUARD_120B_MLX_MXFP4: LLMModelConfig(
            pretrained_model_name="lmstudio-community/gpt-oss-safeguard-120b-MLX-MXFP4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_SAFEGUARD_120B_MLX_MXFP4

    sample_text = "Is this content safe?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="lmstudio-community GPT-OSS Safeguard 120B MLX MXFP4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # MLX MXFP4 quantization is not supported by transformers; strip the
        # quantization_config (which has no quant_method) and use random weights.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model = AutoModelForCausalLM.from_config(config)
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
