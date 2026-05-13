# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PerceiverIO Vision model loader implementation for image classification
"""
import torch
from loguru import logger
from transformers import (
    AutoImageProcessor,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
)
from typing import Optional
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel
from datasets import load_dataset
from ...tools.utils import print_compiled_model_results
import transformers.models.perceiver.modeling_perceiver as pm
from .src.utils import MathShim

pm.np = MathShim()


def _install_dynamo_safe_param_props(module: torch.nn.Module) -> None:
    """Override module.dtype/.device on a single instance to iterate named_parameters().

    Dynamo (torch 2.10.0) has a bug in nn_module.wrap_values that returns an
    undefined ``named_children`` symbol when tracing module.parameters() /
    .children() / .buffers() / .modules(). The named_parameters() branch has
    its own local list and is unaffected.

    Implemented as a per-instance __class__ swap (one fresh subclass per call
    site) so no transformers/torch package state is mutated globally.
    """
    cls = type(module)
    if getattr(cls, "_tt_xla_dynamo_safe", False):
        return

    class _DynamoSafe(cls):
        _tt_xla_dynamo_safe = True

        @property
        def dtype(self):
            return next(
                p.dtype for _, p in self.named_parameters() if p.is_floating_point()
            )

        @property
        def device(self):
            return next(p.device for _, p in self.named_parameters())

    _DynamoSafe.__name__ = cls.__name__
    _DynamoSafe.__qualname__ = cls.__qualname__
    module.__class__ = _DynamoSafe


class ModelVariant(StrEnum):
    """Available PerceiverIO Vision model variants."""

    VISION_PERCEIVER_CONV = "Vision_Perceiver_Conv"
    VISION_PERCEIVER_LEARNED = "Vision_Perceiver_Learned"
    VISION_PERCEIVER_FOURIER = "Vision_Perceiver_Fourier"


class ModelLoader(ForgeModel):
    """PerceiverIO Vision model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.VISION_PERCEIVER_CONV: ModelConfig(
            pretrained_model_name="deepmind/vision-perceiver-conv",
        ),
        ModelVariant.VISION_PERCEIVER_LEARNED: ModelConfig(
            pretrained_model_name="deepmind/vision-perceiver-learned",
        ),
        ModelVariant.VISION_PERCEIVER_FOURIER: ModelConfig(
            pretrained_model_name="deepmind/vision-perceiver-fourier",
        ),
    }

    # Mapping of variants to their corresponding model classes
    _MODEL_CLASSES = {
        ModelVariant.VISION_PERCEIVER_CONV: PerceiverForImageClassificationConvProcessing,
        ModelVariant.VISION_PERCEIVER_LEARNED: PerceiverForImageClassificationLearned,
        ModelVariant.VISION_PERCEIVER_FOURIER: PerceiverForImageClassificationFourier,
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VISION_PERCEIVER_CONV

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PerceiverIO Vision",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.image_processor = None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a PerceiverIO Vision model from HuggingFace."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Get the appropriate model class for this variant
        model_class = self._MODEL_CLASSES[self._variant]

        # Load the model using the appropriate class
        model = model_class.from_pretrained(pretrained_model_name, **kwargs)
        # PerceiverModel.forward calls self.invert_attention_mask -> self.dtype,
        # which triggers a dynamo bug when tracing self.parameters() under
        # torch.compile. Swap the inner perceiver's class to one whose dtype/
        # device properties iterate named_parameters() instead.
        _install_dynamo_safe_param_props(model.perceiver)
        model.eval()

        # Initialize image processor for this variant
        self.image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name, **kwargs
        )

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for PerceiverIO Vision models."""

        if self.image_processor is None:
            raise RuntimeError(
                "Model must be loaded first before loading inputs. Call load_model() first."
            )

        try:
            # Load image from HuggingFace dataset
            dataset = load_dataset("huggingface/cats-image")["test"]
            image = dataset[0]["image"]
            pixel_values = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values
        except Exception as e:
            logger.warning(
                f"Failed to load the image from dataset ({e}), replacing input with random tensor. "
                "Please check if the dataset is available"
            )
            height = self.image_processor.to_dict()["size"]["height"]
            width = self.image_processor.to_dict()["size"]["width"]
            pixel_values = torch.rand(1, 3, height, width).to(torch.float32)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
