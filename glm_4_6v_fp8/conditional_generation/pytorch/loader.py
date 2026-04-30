# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.6V-FP8 model loader implementation for multimodal conditional generation.

The model uses compressed-tensors FP8 quantization (weights stored as float8_e4m3fn).
The stacked expert Parameters (gate_up_proj, down_proj in Glm4vMoeTextNaiveMoe) are
not wrapped by compressed-tensors CompressedLinear, so torch._grouped_mm receives
raw FP8 tensors. This is unsupported on CPU and TT device. We dequantize those
Parameters to bfloat16 after loading so the default grouped_mm path can proceed.

Note: the full model is ~100 GB in FP8 and ~200 GB in BF16, far exceeding n150's
~12 GB single-device DRAM. This test is KNOWN_FAILURE_XFAIL (hardware-class).
"""
import torch
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


def _dequantize_fp8_experts(model, dtype=torch.bfloat16):
    # The Glm4vMoeTextNaiveMoe stores gate_up_proj and down_proj as stacked
    # nn.Parameters in float8_e4m3fn. torch._grouped_mm requires float32/bf16/fp16.
    # Cast to bf16 so the default grouped_mm dispatch works on CPU and TT.
    for module in model.modules():
        if isinstance(module, Glm4vMoeTextNaiveMoe):
            if module.gate_up_proj.dtype == torch.float8_e4m3fn:
                module.gate_up_proj.data = module.gate_up_proj.data.to(dtype)
            if module.down_proj.dtype == torch.float8_e4m3fn:
                module.down_proj.data = module.down_proj.data.to(dtype)


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
        _dequantize_fp8_experts(model, dtype=dtype_override or torch.bfloat16)
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
