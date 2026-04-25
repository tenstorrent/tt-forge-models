# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QoQ-Med-Omni-7B model loader implementation for image to text.
"""
import sys
import torch
from torch.overrides import TorchFunctionMode
from transformers import AutoModel, AutoProcessor, AutoConfig
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


class _VisualWrapper:
    """Wraps Qwen2_5_VisionTransformerPretrainedModel for transformers 5.x compatibility.

    In transformers 5.x, the visual model returns BaseModelOutputWithPooling, but
    the custom TimeSeriesQwen2_5_VL forward code expects a raw tensor (pooler_output).
    """

    def __init__(self, visual):
        self._visual = visual

    @property
    def dtype(self):
        return self._visual.dtype

    def __call__(self, pixel_values, **kwargs):
        output = self._visual(pixel_values, **kwargs)
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        return output

    def __getattr__(self, name):
        return getattr(self._visual, name)


class _LinspaceCpuMode(TorchFunctionMode):
    """Force torch.linspace to CPU device during model init on meta device.

    Transformers 5.x always initializes models on the meta device. The custom
    TimeSeriesQwen2_5_VL code calls torch.linspace(...).item() during __init__,
    which fails on meta tensors. This mode overrides device='cpu' for linspace
    calls so .item() succeeds while the rest of the model initializes on meta.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.linspace:
            kwargs["device"] = "cpu"
        return func(*args, **kwargs)


class ModelVariant(StrEnum):
    """Available QoQ-Med-Omni-7B model variants for image to text."""

    QOQ_MED_OMNI_7B = "7b"


class ModelLoader(ForgeModel):
    """QoQ-Med-Omni-7B model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QOQ_MED_OMNI_7B: LLMModelConfig(
            pretrained_model_name="ddvd233/QoQ-Med-Omni-7B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QOQ_MED_OMNI_7B

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

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
            model="QoQ-Med-Omni-7B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the QoQ-Med-Omni-7B model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The QoQ-Med-Omni-7B model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": False,
            "device_map": None,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        # The custom TimeSeriesQwen2_5_VL model code accesses config.hidden_size directly,
        # but Qwen2_5_VLConfig stores it on text_config. Patch it here as a workaround.
        if not hasattr(config, "hidden_size"):
            config.hidden_size = config.get_text_config().hidden_size
        model_kwargs["config"] = config

        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        with _LinspaceCpuMode():
            model = AutoModel.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()

        # Patch the custom model class to fix transformers 5.x API changes:
        # 1. self.visual moved from ForConditionalGeneration to self.model.visual
        # 2. self.visual() now returns BaseModelOutputWithPooling instead of a tensor
        if hasattr(model, "model") and hasattr(model.model, "visual"):
            visual_wrapper = _VisualWrapper(model.model.visual)
            type(model).visual = property(lambda self, _w=visual_wrapper: _w)

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the QoQ-Med-Omni-7B model.

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
