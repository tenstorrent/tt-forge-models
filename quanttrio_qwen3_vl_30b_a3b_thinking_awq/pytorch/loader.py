# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QuantTrio Qwen3-VL-30B-A3B-Thinking AWQ model loader implementation for image to text.
"""

import gptqmodel  # noqa: F401 — must import before transformers enters init_empty_weights context
from transformers import (
    AutoConfig,
    Qwen3VLMoeForConditionalGeneration,
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

        # Fix modules_to_not_convert: the model config uses bare names like 'visual'
        # but named_modules() returns 'model.visual.*', so the prefix matching fails and
        # gptqmodel incorrectly replaces vision MLP layers (intermediate_size=4304,
        # 4304 % 32 != 0) causing an assertion error in convert_weight_packed_zp.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        qc = getattr(config, "quantization_config", None)
        if qc is not None:
            modules_key = "modules_to_not_convert"
            modules = (
                qc.get(modules_key)
                if isinstance(qc, dict)
                else getattr(qc, modules_key, None)
            )
            if modules:
                fixed = [
                    f"model.{m}" if not m.startswith("model.") else m for m in modules
                ]
                if isinstance(qc, dict):
                    qc[modules_key] = fixed
                else:
                    setattr(qc, modules_key, fixed)
        model_kwargs["config"] = config

        # AWQ repos may not ship a processor; fall back to the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Thinking")

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

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
