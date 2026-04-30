# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.6V-FP8 model loader implementation for multimodal conditional generation.

The model uses compressed-tensors FP8 quantization (weights stored as float8_e4m3fn,
activations dynamically quantized at runtime). The default experts implementation
(grouped_mm / eager) calls matmul with float8 matrices, which is unsupported on
CPU and TT device. We post-load-patch each Glm4vMoeTextNaiveMoe.forward with a
static per-expert masked matmul that casts FP8 weights to the model dtype (bfloat16)
before each F.linear call. This bypasses the @use_experts_implementation dispatch,
which validates implementation names against a fixed allowlist.
"""
import types
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration
from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextNaiveMoe
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
from ....tools.utils import cast_input_to_type, get_file
from PIL import Image


def _static_fp8_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    # Static per-expert masked matmul with explicit FP8→dtype cast on weights.
    # Avoids torch._grouped_mm / F.linear with float8 (unsupported on CPU/TT).
    # Also avoids torch.nonzero / torch.where (dynamic shapes on TT XLA).
    dtype = hidden_states.dtype
    out = torch.zeros_like(hidden_states)
    for expert_idx in range(self.num_experts):
        mask = (top_k_index == expert_idx)  # [tokens, top_k]
        weight = (mask.to(dtype) * top_k_weights.to(dtype)).sum(dim=-1, keepdim=True)
        gate_up = F.linear(hidden_states, self.gate_up_proj[expert_idx].to(dtype))
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_expert = F.linear(
            self.act_fn(gate) * up, self.down_proj[expert_idx].to(dtype)
        )
        out = out + hidden_expert * weight
    return out.to(dtype)


def _patch_experts(model):
    for module in model.modules():
        if isinstance(module, Glm4vMoeTextNaiveMoe):
            module.forward = types.MethodType(_static_fp8_experts_forward, module)


class ModelVariant(StrEnum):
    """Available GLM-4.6V-FP8 model variants for multimodal conditional generation."""

    GLM_4_6V_FP8 = "glm_4_6v_fp8"


class ModelLoader(ForgeModel):
    """GLM-4.6V-FP8 model loader implementation for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.GLM_4_6V_FP8: LLMModelConfig(
            pretrained_model_name="zai-org/GLM-4.6V-FP8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_6V_FP8

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="glm_4_6v_fp8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, use_fast=False, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Glm4vMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        _patch_experts(model)
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
