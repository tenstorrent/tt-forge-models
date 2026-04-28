# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
btbtyler09/Qwen3-Coder-Next-GPTQ-4bit model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTQConfig
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
    """Available btbtyler09/Qwen3-Coder-Next-GPTQ-4bit model variants for causal language modeling."""

    QWEN3_CODER_NEXT_GPTQ_4BIT = "Qwen3-Coder-Next-GPTQ-4bit"


class ModelLoader(ForgeModel):
    """btbtyler09/Qwen3-Coder-Next-GPTQ-4bit model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_CODER_NEXT_GPTQ_4BIT: LLMModelConfig(
            pretrained_model_name="btbtyler09/Qwen3-Coder-Next-GPTQ-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_CODER_NEXT_GPTQ_4BIT

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
            model="btbtyler09-Qwen3-Coder-Next-GPTQ-4bit",
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

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
            else:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        # The default GPTQ backend (EXLLAMA/HF_KERNEL) attempts to load CUDA
        # extensions that segfault on machines with TT hardware but no NVIDIA GPU.
        # Force GPTQ_TORCH, a pure-PyTorch backend that requires no CUDA.
        model_kwargs["quantization_config"] = GPTQConfig(bits=4, backend="gptq_torch")

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Dequantize GPTQ int4 weights to float tensors. GPTQ_TORCH QuantLinear
        # uses boolean-mask indexing that produces dynamic shapes incompatible with
        # XLA static-shape compilation. Replace each TorchQuantLinear with a plain
        # nn.Linear. Build the module map once (O(N)) to avoid the O(N²) cost of
        # rebuilding dict(model.named_modules()) inside the loop as done in
        # gptqmodel.dequantize_model.
        import torch.nn as nn
        from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

        module_map = dict(model.named_modules())
        for name, mod in list(module_map.items()):
            if not isinstance(mod, TorchQuantLinear):
                continue
            dq = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            dq.weight = nn.Parameter(mod.dequantize_weight().T.detach())
            if mod.bias is not None:
                dq.bias = nn.Parameter(mod.bias.detach())
            if dtype_override is not None:
                dq = dq.to(dtype_override)
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                setattr(module_map[parent_name], child_name, dq)
            else:
                setattr(model, name, dq)
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config

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
