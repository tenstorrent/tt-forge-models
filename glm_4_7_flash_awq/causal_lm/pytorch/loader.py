# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash AWQ model loader implementation for causal language modeling.

AWQ-quantized variants of GLM-4.7-Flash from TheHouseOfTheDude, published on
separate branches of the HuggingFace repository. Both variants use the
compressed-tensors runtime format with MoE-aware calibration.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
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


def _tt_static_glm4_moe_lite_forward(experts_module, hidden_states, top_k_index, top_k_weights):
    # Per-expert masked matmul. Avoids grouped_mm (histc on Int, unsupported on XLA)
    # and batched_mm (3D dynamic gather, L1 CB overflow).
    # Loop over Python ints so dynamo unrolls into static F.linear calls.
    dtype = hidden_states.dtype
    out = torch.zeros_like(hidden_states)
    for expert_idx in range(experts_module.num_experts):
        mask = (top_k_index == expert_idx)  # [tokens, top_k]
        weight = (mask.to(dtype) * top_k_weights.to(dtype)).sum(dim=-1, keepdim=True)
        gate_up = F.linear(hidden_states, experts_module.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_expert = F.linear(
            experts_module.act_fn(gate) * up, experts_module.down_proj[expert_idx]
        )
        out = out + hidden_expert * weight
    return out.to(dtype)


ALL_EXPERTS_FUNCTIONS["tt_static_glm4_moe_lite"] = _tt_static_glm4_moe_lite_forward


class ModelVariant(StrEnum):
    """Available GLM-4.7-Flash AWQ model variants for causal language modeling."""

    W4A16_GS32 = "W4A16_GS32"
    W8A16_GS32 = "W8A16_GS32"


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash AWQ model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.W4A16_GS32: LLMModelConfig(
            pretrained_model_name="TheHouseOfTheDude/GLM-4.7-Flash_AWQ",
            max_length=128,
        ),
        ModelVariant.W8A16_GS32: LLMModelConfig(
            pretrained_model_name="TheHouseOfTheDude/GLM-4.7-Flash_AWQ",
            max_length=128,
        ),
    }

    # Map variant enum to HuggingFace revision branch
    _REVISION_MAP = {
        ModelVariant.W4A16_GS32: "W4A16_GS32",
        ModelVariant.W8A16_GS32: "W8A16_GS32",
    }

    DEFAULT_VARIANT = ModelVariant.W4A16_GS32

    sample_text = "Hey how are you doing today?"

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
            model="GLM-4.7-Flash AWQ",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=self._REVISION_MAP[self._variant],
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        revision = self._REVISION_MAP[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, revision=revision, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, revision=revision, **model_kwargs
        ).eval()

        # Override grouped_mm dispatch (histc on Int unsupported on XLA) with
        # a static per-expert loop that dynamo can unroll.
        model.config._experts_implementation = "tt_static_glm4_moe_lite"

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=self._REVISION_MAP[self._variant],
            trust_remote_code=True,
        )
        return self.config
