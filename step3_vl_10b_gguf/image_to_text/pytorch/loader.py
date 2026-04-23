# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Step3 VL 10B GGUF model loader implementation for image to text.
"""
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
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
    """Available Step3 VL 10B GGUF model variants for image to text."""

    STEP3_VL_10B_GGUF_Q4_K_M = "10b_gguf_q4_k_m"


class ModelLoader(ForgeModel):
    """Step3 VL 10B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M: LLMModelConfig(
            pretrained_model_name="seanbailey518/Step3-VL-10B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M: "Step3-VL-10B-Q4_K_M.gguf",
    }

    # Processor is loaded from the original non-GGUF repo since the GGUF repo
    # only contains quantized model weights without tokenizer/processor configs.
    _PROCESSOR_SOURCE = "stepfun-ai/Step3-VL-10B"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    sample_text = "Describe this image."

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
            model="Step3 VL 10B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._PROCESSOR_SOURCE, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Step3 VL 10B GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Step3 VL 10B GGUF model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load config from the original non-GGUF repo to get the Step3-VL model class
        # (AutoModelForCausalLM -> Step3VL10BForCausalLM via remote code).
        # The GGUF file metadata reports Qwen3ForCausalLM which doesn't match the
        # VL model class, so we must override with the correct config.
        config = AutoConfig.from_pretrained(
            self._PROCESSOR_SOURCE,
            trust_remote_code=True,
        )

        if self.num_layers is not None:
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
            else:
                config.num_hidden_layers = self.num_layers

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file

        if self.processor is None:
            self._load_processor()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Step3 VL 10B GGUF model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
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
        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._PROCESSOR_SOURCE,
            trust_remote_code=True,
        )
        return self.config
