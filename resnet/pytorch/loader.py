# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from torchvision import models
import torch
import timm

from transformers import ResNetForImageClassification
from ...tools.utils import VisionPreprocessor, VisionPostprocessor


from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


@dataclass
class ResNetConfig(ModelConfig):
    """Configuration specific to ResNet models"""

    source: ModelSource
    high_res_size: tuple = (
        None  # None means use default size, otherwise (width, height)
    )


class ModelVariant(StrEnum):
    """Available ResNet model variants."""

    # HuggingFace variants
    RESNET_50_HF = "resnet_50_hf"
    RESNET_50_HF_HIGH_RES = "resnet_50_hf_high_res"

    # TIMM variants
    RESNET_50_TIMM = "resnet50_timm"
    RESNET_50_TIMM_HIGH_RES = "resnet50_timm_high_res"

    # Torchvision variants
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_50_HIGH_RES = "resnet50_high_res"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"


class ModelLoader(ForgeModel):
    """ResNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # HuggingFace variants
        ModelVariant.RESNET_50_HF: ResNetConfig(
            pretrained_model_name="microsoft/resnet-50",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.RESNET_50_HF_HIGH_RES: ResNetConfig(
            pretrained_model_name="microsoft/resnet-50",
            source=ModelSource.HUGGING_FACE,
            high_res_size=(1280, 800),
        ),
        # TIMM variants
        ModelVariant.RESNET_50_TIMM: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TIMM,
        ),
        ModelVariant.RESNET_50_TIMM_HIGH_RES: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TIMM,
            high_res_size=(1280, 800),
        ),
        # Torchvision variants
        ModelVariant.RESNET_18: ResNetConfig(
            pretrained_model_name="resnet18",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.RESNET_34: ResNetConfig(
            pretrained_model_name="resnet34",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.RESNET_50: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.RESNET_50_HIGH_RES: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TORCHVISION,
            high_res_size=(1280, 800),
        ),
        ModelVariant.RESNET_101: ResNetConfig(
            pretrained_model_name="resnet101",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.RESNET_152: ResNetConfig(
            pretrained_model_name="resnet152",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNET_50_HF

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        if variant in [
            ModelVariant.RESNET_50_HF_HIGH_RES,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="resnet",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the ResNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ResNet model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            # Load model from HuggingFace
            model = ResNetForImageClassification.from_pretrained(model_name)

        elif source == ModelSource.TIMM:
            # Load model using timm
            model = timm.create_model(model_name, pretrained=True)

        elif source == ModelSource.TORCHVISION:
            # Load model from torchvision
            # Get the weights class name (e.g., "resnet50" -> "ResNet50_Weights")
            weight_class_name = model_name.replace("resnet", "ResNet") + "_Weights"

            # Get the weights class and model function
            weights = getattr(models, weight_class_name).DEFAULT
            model_func = getattr(models, model_name)
            model = model_func(weights=weights)

        model.eval()

        # Store model for potential use in input preprocessing and postprocessing
        self.model = model
        
        # Update preprocessor with cached model (for TIMM models)
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)
        
        # Update postprocessor with model instance (for HuggingFace models)
        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def _get_preprocessor(self):
        """Get or create the vision preprocessor for this instance.
        
        Returns:
            VisionPreprocessor: Configured preprocessor instance
        """
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source
            high_res_size = self._variant_config.high_res_size
            
            # Define weight class name transformation for torchvision
            def weight_class_name_fn(name: str) -> str:
                return name.replace("resnet", "ResNet") + "_Weights"
            
            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
                high_res_size=high_res_size,
                weight_class_name_fn=weight_class_name_fn if source == ModelSource.TORCHVISION else None,
            )
            
            # Set cached model if available (for TIMM models)
            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)
        
        return self._preprocessor

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the ResNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.
            image: Optional input image. Can be:
                  - PIL.Image.Image: A PIL Image object to preprocess
                  - str: URL string to download and load image from
                  - torch.Tensor: Pre-processed tensor (will be used as-is after batch replication)
                  - List[Union[Image.Image, str]]: List of PIL Images or URLs for batched evaluation
                  - None: Uses default sample image from COCO dataset

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for ResNet.
        """
        preprocessor = self._get_preprocessor()
        
        # For TIMM models, we may need the model for config
        model_for_config = None
        if self._variant_config.source == ModelSource.TIMM:
            if hasattr(self, "model") and self.model is not None:
                model_for_config = self.model
        
        return preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

    def _get_postprocessor(self):
        """Get or create the vision postprocessor for this instance.
        
        Returns:
            VisionPostprocessor: Configured postprocessor instance
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source
            
            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )
        
        return self._postprocessor

    def post_process(
        self,
        output=None,
        co_out=None,
        framework_model=None,
        compiled_model=None,
        inputs=None,
        dtype_override=None,
    ):
        """Post-processes model outputs based on the model source.

        This method can be used in two ways:
        1. With 'output' parameter: Returns a dictionary with prediction information
        2. With legacy parameters (co_out, framework_model, etc.): Prints results (backward compatibility)

        Args:
            output: Model output tensor. Can be:
                   - torch.Tensor: Raw logits from model forward pass
                   - For HuggingFace models: Can be a ModelOutput object with logits attribute
                   - If provided, returns dict with label and probability
            co_out: Outputs from the compiled model (legacy parameter, used for printing)
            framework_model: The original framework-based model (legacy parameter)
            compiled_model: The compiled version of the model (legacy parameter)
            inputs: A list of images to process and classify (legacy parameter)
            dtype_override: Optional torch.dtype to override the input's dtype (legacy parameter)

        Returns:
            dict or None: If 'output' is provided, returns dictionary with:
                          {
                              "label": str,  # Top-1 predicted class label
                              "probability": str  # Top-1 probability as percentage string (e.g., "98.34%")
                          }
                          Otherwise, prints results and returns None (backward compatibility).
        """
        postprocessor = self._get_postprocessor()
        
        # New usage: return dict from output tensor
        if output is not None:
            return postprocessor.postprocess(output, top_k=1, return_dict=True)

        # Legacy usage: print results (backward compatibility)
        postprocessor.print_results(
            co_out=co_out,
            framework_model=framework_model,
            compiled_model=compiled_model,
            inputs=inputs,
            dtype_override=dtype_override,
        )
        return None