# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/SpaceThinker-Qwen2.5VL-3B-GGUF model loader implementation for image to text.
"""

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
    """Available mradermacher SpaceThinker-Qwen2.5VL-3B-GGUF variants for image to text."""

    SPACETHINKER_QWEN2_5_VL_3B_Q4_K_M_GGUF = "SpaceThinker_Qwen2_5_VL_3B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher SpaceThinker-Qwen2.5VL-3B-GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.SPACETHINKER_QWEN2_5_VL_3B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/SpaceThinker-Qwen2.5VL-3B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPACETHINKER_QWEN2_5_VL_3B_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.SPACETHINKER_QWEN2_5_VL_3B_Q4_K_M_GGUF: "SpaceThinker-Qwen2.5VL-3B.Q4_K_M.gguf",
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
        return ModelInfo(
            model="mradermacher SpaceThinker-Qwen2.5VL-3B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SpaceThinker-Qwen2.5VL-3B-GGUF model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SpaceThinker-Qwen2.5VL-3B-GGUF model."""
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
