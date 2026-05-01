# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/granite-4.0-h-tiny-4bit model loader implementation for causal language modeling.

mlx-community/granite-4.0-h-tiny-4bit is an MLX-quantized (4-bit) derivative of
ibm-granite/granite-4.0-h-tiny, a hybrid Mamba2 + MoE causal language model from
the Granite 4.0 family. It is exposed as a Granite 4.0 Hybrid causal LM via
Hugging Face Transformers.
"""
import types
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


def _patched_topk_gating_forward(self, hidden_states):
    """Avoids expert_size.tolist() by returning sorted_expert_ids as a tensor.

    GraniteMoeHybridTopKGating.forward calls expert_size.tolist() which
    triggers a device-to-host transfer that fails on TT silicon with
    INTERNAL error code 13.
    """
    logits = self.layer(hidden_states).float()
    top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
    top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)

    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")

    sorted_expert_ids = top_k_experts[index_sorted_experts].int()

    top_k_gates = top_k_gates.flatten()
    batch_gates = top_k_gates[index_sorted_experts]

    return index_sorted_experts, batch_index, batch_gates, sorted_expert_ids, logits


def _patched_parallel_experts_forward(self, inputs, sorted_expert_ids):
    """Per-expert masked matmul instead of split-by-expert-size.

    The original inputs.split(expert_size) requires expert_size as a Python
    list (D2H transfer). A weight gather also fails: MLIR flattens the 3D
    weight to a 2D embedding table whose row size (~3 MB) overflows L1.
    Per-expert static loop avoids all device-to-host transfers.
    """
    T = inputs.shape[0]
    result = torch.zeros(T, self.output_size, dtype=inputs.dtype, device=inputs.device)
    for e in range(self.num_experts):
        w_e = self.weight[e]  # [output_size, input_size] — static slice
        out_e = torch.nn.functional.linear(inputs, w_e)  # [T, output_size]
        mask_e = (sorted_expert_ids == e).to(inputs.dtype).unsqueeze(1)
        result = result + out_e * mask_e
    return result


def _patch_moe_experts(model):
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridParallelExperts,
        GraniteMoeHybridTopKGating,
    )

    for module in model.modules():
        if isinstance(module, GraniteMoeHybridTopKGating):
            module.forward = types.MethodType(_patched_topk_gating_forward, module)
        elif isinstance(module, GraniteMoeHybridParallelExperts):
            module.forward = types.MethodType(
                _patched_parallel_experts_forward, module
            )


class ModelVariant(StrEnum):
    """Available mlx-community/granite-4.0-h-tiny-4bit model variants."""

    GRANITE_4_0_H_TINY_4BIT = "granite-4.0-h-tiny-4bit"


class ModelLoader(ForgeModel):
    """mlx-community/granite-4.0-h-tiny-4bit model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_H_TINY_4BIT: LLMModelConfig(
            # MLX 4-bit weights are packed in a format incompatible with standard
            # PyTorch/transformers; load the base model which has the same architecture.
            pretrained_model_name="ibm-granite/granite-4.0-h-tiny",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_H_TINY_4BIT

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
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="granite-4.0-h-tiny-4bit",
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
            pretrained_model_name, **tokenizer_kwargs
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        _patch_moe_experts(model)

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
            self._variant_config.pretrained_model_name,
        )
        return self.config
