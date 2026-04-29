# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
cyankiwi Qwen3-VL-8B-Instruct AWQ 8-bit model loader implementation for image to text.
"""

from transformers import (
    AutoConfig,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

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
    """Available cyankiwi Qwen3-VL-8B-Instruct AWQ 8-bit model variants."""

    QWEN3_VL_8B_INSTRUCT_AWQ_8BIT = "8b_instruct_awq_8bit"


class ModelLoader(ForgeModel):
    """cyankiwi Qwen3-VL-8B-Instruct AWQ 8-bit model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_8B_INSTRUCT_AWQ_8BIT: LLMModelConfig(
            pretrained_model_name="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-8bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_8B_INSTRUCT_AWQ_8BIT

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
            model="cyankiwi_qwen3_vl_8b_instruct_awq_8bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the cyankiwi Qwen3-VL-8B-Instruct AWQ 8-bit model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The cyankiwi Qwen3-VL-8B-Instruct AWQ 8-bit model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = "auto"
            model_kwargs["device_map"] = "auto"

        # TT hardware has no int8 matmul path; dequantize compressed-tensors
        # pack-quantized weights to bfloat16 at load time.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        qc = getattr(config, "quantization_config", None)
        if isinstance(qc, dict):
            qc["run_compressed"] = False
        elif qc is not None:
            qc.run_compressed = False
        model_kwargs["config"] = config

        model_kwargs |= kwargs

        # AWQ repos may not ship a processor; fall back to the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # compressed-tensors 0.15.x attaches instance-level forward overrides to
        # quantized Linears that access weight.data unconditionally, conflicting
        # with TT-XLA's __torch_function__. Remove them to restore class-level forward.
        for m in model.modules():
            if "forward" in m.__dict__:
                del m.__dict__["forward"]

        model.config.use_cache = False
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the cyankiwi Qwen3-VL-8B-Instruct AWQ 8-bit model.

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
