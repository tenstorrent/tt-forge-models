# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Magistral Small MLX model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration,
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

# Mistral3Config is not registered with AutoModelForCausalLM (it's VLM, registered for
# image-text-to-text), but Magistral Small is a text-only reasoning model using the same
# architecture. Register it so from_pretrained resolves correctly.
if Mistral3Config not in AutoModelForCausalLM._model_mapping._extra_content:
    AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration)


class ModelVariant(StrEnum):
    """Available Magistral Small MLX model variants for causal language modeling."""

    SMALL_2509_MLX_5BIT = "Small_2509_MLX_5bit"


class ModelLoader(ForgeModel):
    """Magistral Small MLX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SMALL_2509_MLX_5BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Magistral-Small-2509-MLX-5bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_2509_MLX_5BIT

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
            model="Magistral Small MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
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

        # transformers 5.x rejects MLX quantization configs (no quant_method); strip it.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config") and not hasattr(
            config.quantization_config, "quant_method"
        ):
            del config.quantization_config
        model_kwargs["config"] = config
        # MLX quantized weights have different shapes (packed uint32 vs float); reinit mismatches.
        model_kwargs["ignore_mismatched_sizes"] = True

        if self.num_layers is not None:
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
            else:
                config.num_hidden_layers = self.num_layers

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Initialize any meta tensors left by missing vision tower weights.
        target_dtype = dtype_override or torch.bfloat16
        for name, param in model.named_parameters():
            if param.is_meta:
                new_param = torch.nn.Parameter(
                    torch.empty(param.shape, dtype=target_dtype, device="cpu").normal_(
                        std=0.02
                    )
                )
                parts = name.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                setattr(mod, parts[-1], new_param)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
