# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.6V-FP8 model loader implementation for multimodal conditional generation.

The model uses compressed-tensors FP8 quantization (weights stored as float8_e4m3fn,
activations dynamically quantized at runtime). The default experts implementation
(grouped_mm) calls torch._grouped_mm with float8 matrices, which is unsupported
on CPU and TT device. We register a static per-expert forward that casts FP8
weights to the model dtype (bfloat16) before each matmul.
"""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor, Glm4vMoeForConditionalGeneration
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
from ....tools.utils import cast_input_to_type, get_file
from PIL import Image


def _tt_static_glm4v_fp8_moe_forward(experts_module, hidden_states, top_k_index, top_k_weights):
    # Per-expert masked matmul with explicit FP8→dtype cast on weights.
    # Avoids torch._grouped_mm (unsupported for float8 on CPU/TT) and
    # torch.histc-on-int (grouped_mm path). Mirrors the GLM-4.7 AWQ pattern.
    dtype = hidden_states.dtype
    out = torch.zeros_like(hidden_states)
    for expert_idx in range(experts_module.num_experts):
        mask = (top_k_index == expert_idx)  # [tokens, top_k]
        weight = (mask.to(dtype) * top_k_weights.to(dtype)).sum(dim=-1, keepdim=True)
        gate_up = F.linear(
            hidden_states, experts_module.gate_up_proj[expert_idx].to(dtype)
        )
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_expert = F.linear(
            experts_module.act_fn(gate) * up,
            experts_module.down_proj[expert_idx].to(dtype),
        )
        out = out + hidden_expert * weight
    return out.to(dtype)


ALL_EXPERTS_FUNCTIONS["tt_static_glm4v_fp8_moe"] = _tt_static_glm4v_fp8_moe_forward


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

        config = AutoConfig.from_pretrained(pretrained_model_name)
        config._experts_implementation = "tt_static_glm4v_fp8_moe"

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Glm4vMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
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
