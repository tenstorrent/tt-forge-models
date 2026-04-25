# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QuantTrio Qwen3-VL-30B-A3B-Thinking AWQ model loader implementation for image to text.
"""

import gptqmodel  # noqa: F401 — must import before from_pretrained enters meta-device context
import gptqmodel.nn_modules.qlinear.gemm_hf_kernel_awq as _awq_hf_kernel
import gptqmodel.quantization.awq.utils.packing_utils as _awq_packing
from gptqmodel.nn_modules.qlinear.gemm_hf_kernel_awq import HFKernelAwqLinear

from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

import torch

# Patch dequantize_gemm to crop padded scales/zeros to actual weight size.
# Some AWQ checkpoints pad in_features to the next group_size boundary; the
# original function assumes exact alignment and errors with a size mismatch.
_orig_dequantize_gemm = _awq_packing.dequantize_gemm


def _dequantize_gemm_padded(qweight, qzeros, scales, bits, group_size):
    iweight, izeros = _awq_packing.unpack_awq(qweight, qzeros, bits)
    iweight, izeros = _awq_packing.reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    K = iweight.shape[0]
    scales_exp = scales.repeat_interleave(group_size, dim=0)[:K]
    izeros_exp = izeros.repeat_interleave(group_size, dim=0)[:K]
    return (iweight - izeros_exp) * scales_exp


_awq_packing.dequantize_gemm = _dequantize_gemm_padded
_awq_hf_kernel.dequantize_gemm = _dequantize_gemm_padded

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available QuantTrio Qwen3-VL-30B-A3B-Thinking AWQ model variants."""

    QWEN3_VL_30B_A3B_THINKING_AWQ = "30b_a3b_thinking_awq"


class ModelLoader(ForgeModel):
    """QuantTrio Qwen3-VL-30B-A3B-Thinking AWQ model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_30B_A3B_THINKING_AWQ: LLMModelConfig(
            pretrained_model_name="QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_30B_A3B_THINKING_AWQ

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="quanttrio_qwen3_vl_30b_a3b_thinking_awq",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the QuantTrio Qwen3-VL-30B-A3B-Thinking AWQ model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The QuantTrio Qwen3-VL-30B-A3B-Thinking AWQ model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = "auto"
            model_kwargs["device_map"] = "auto"

        model_kwargs |= kwargs

        # AWQ repos may not ship a processor; fall back to the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Thinking")

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # gptqmodel's optimized CPU AWQ kernel asserts N % 32 == 0, which fails
        # for some MoE expert weight shapes in this model. Force the dequantize
        # fallback path by pre-setting linear_mode so transform() is never called.
        for module in model.modules():
            if isinstance(module, HFKernelAwqLinear):
                module.linear_mode = "train"

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the QuantTrio Qwen3-VL-30B-A3B-Thinking AWQ model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
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
        return inputs
