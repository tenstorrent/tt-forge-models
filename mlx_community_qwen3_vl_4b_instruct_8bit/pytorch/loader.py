# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/Qwen3-VL-4B-Instruct-8bit model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
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
    """Available mlx-community Qwen3-VL-4B-Instruct-8bit model variants."""

    QWEN3_VL_4B_INSTRUCT_8BIT = "4b_instruct_8bit"


class ModelLoader(ForgeModel):
    """mlx-community Qwen3-VL-4B-Instruct-8bit model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_4B_INSTRUCT_8BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/Qwen3-VL-4B-Instruct-8bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_4B_INSTRUCT_8BIT

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
            model="mlx_community_qwen3_vl_4b_instruct_8bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the mlx-community Qwen3-VL-4B-Instruct-8bit model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = "auto"
            model_kwargs["device_map"] = "auto"

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        # The MLX-community 8bit model has an MLX-style quantization_config
        # (bits/group_size/mode) that lacks the `quant_method` field required
        # by transformers. Delete the attribute so transformers skips quantization.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            del config.quantization_config

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

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
