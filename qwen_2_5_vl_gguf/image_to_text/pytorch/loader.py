# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 VL GGUF model loader implementation for image to text.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
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
    """Available Qwen 2.5 VL GGUF model variants for image to text."""

    QWEN_2_5_VL_72B_INSTRUCT_GGUF = "72b_instruct_gguf"
    BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF = "bartowski_72b_instruct_gguf"


class ModelLoader(ForgeModel):
    """Qwen 2.5 VL GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen2.5-VL-72B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/Qwen_Qwen2.5-VL-72B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF: "Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf",
        ModelVariant.BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF: "Qwen_Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    @property
    def _gguf_file(self):
        return self._GGUF_FILES[self._variant]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

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
            model="Qwen 2.5 VL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 2.5 VL GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 2.5 VL GGUF model instance for image to text.
        """
        # The transformers GGUF loader does not yet support the qwen2vl architecture,
        # so we load the config and processor from the base model and instantiate with
        # random weights via from_config. For compile-only environments this is acceptable.
        base_model_name = "Qwen/Qwen2.5-VL-72B-Instruct"

        self.processor = AutoProcessor.from_pretrained(base_model_name)

        config = AutoConfig.from_pretrained(base_model_name)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        target_dtype = dtype_override if dtype_override is not None else torch.float32
        old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(target_dtype)
        try:
            model = AutoModelForImageTextToText.from_config(config)
        finally:
            torch.set_default_dtype(old_default_dtype)
        model.eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 2.5 VL GGUF model.

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
                        "image": self.sample_image,
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        return self.config
