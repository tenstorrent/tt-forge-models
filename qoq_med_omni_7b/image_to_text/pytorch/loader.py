# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QoQ-Med-Omni-7B model loader implementation for image to text.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
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
    """Wraps Qwen2_5_VisionTransformerPretrainedModel so that __call__ returns
    the last_hidden_state tensor instead of BaseModelOutputWithPooling.
    In transformers 5.x the visual encoder return type changed from a raw
    tensor to a named-tuple output; the custom model code still expects a tensor."""

    def __init__(self, visual):
        self._visual = visual

    @property
    def dtype(self):
        return self._visual.dtype

    def __call__(self, *args, **kwargs):
        out = self._visual(*args, **kwargs)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out

    def __getattr__(self, name):
        return getattr(self._visual, name)


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

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        # hidden_size is not a direct attribute in transformers 5.x; patch it so
        # the custom model code (config.hidden_size) doesn't raise AttributeError.
        if not hasattr(config, "hidden_size"):
            config.hidden_size = config.get_text_config().hidden_size

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        # AutoModelForImageTextToText has no mapping for the custom
        # TimeSeriesQwen2_5_VLConfig; register it before loading.
        model_cls_str = (
            config.auto_map.get("AutoModelForImageTextToText")
            or config.auto_map.get("AutoModelForVision2Seq")
            or config.auto_map.get("AutoModel")
        )
        model_class = get_class_from_dynamic_module(
            model_cls_str, pretrained_model_name
        )
        AutoModelForImageTextToText.register(type(config), model_class, exist_ok=True)

        # In transformers 5.x, Qwen2_5_VLForConditionalGeneration moved `visual`
        # into self.model.visual; the custom forward still accesses self.visual.
        # Also wrap it so __call__ returns a tensor (last_hidden_state) instead
        # of BaseModelOutputWithPooling which the custom forward doesn't handle.
        if not hasattr(model_class, "visual"):
            model_class.visual = property(
                lambda self: _VisualWrapper(self.model.visual)
            )

        model_kwargs["config"] = config

        # transformers 5.x wraps model __init__ in torch.device("meta"), but
        # the custom time-series module calls torch.linspace(...).item() which
        # fails on meta tensors. Wrap TimeSeriesEmbedding.__init__ in
        # torch.device("cpu") to force all sub-modules to CPU.
        TimeSeriesEmbedding = get_class_from_dynamic_module(
            "modeling_time_series_qwen2_5_vl.TimeSeriesEmbedding",
            pretrained_model_name,
        )
        _orig_ts_embedding_init = TimeSeriesEmbedding.__init__

        def _patched_ts_embedding_init(self, *args, **kwargs):
            with torch.device("cpu"):
                _orig_ts_embedding_init(self, *args, **kwargs)

        TimeSeriesEmbedding.__init__ = _patched_ts_embedding_init

        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

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
