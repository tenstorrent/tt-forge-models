# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DLA model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from datasets import load_dataset

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
from ...tools.utils import print_compiled_model_results


@dataclass
class DLAConfig(ModelConfig):
    """Configuration specific to DLA models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available DLA model variants (all loaded via timm / HuggingFace Hub)."""

    DLA34 = "34"
    DLA46_C = "46_C"
    DLA46X_C = "46x_C"
    DLA60 = "60"
    DLA60X = "60x"
    DLA60X_C = "60x_C"
    DLA102 = "102"
    DLA102X = "102x"
    DLA102X2 = "102x2"
    DLA169 = "169"


class ModelLoader(ForgeModel):
    """DLA model loader implementation."""

    # Dictionary of available model variants using structured configs.
    # All variants use timm (HuggingFace Hub) — the original dl.yf.io host
    # has been permanently down since 2024.
    _VARIANTS = {
        ModelVariant.DLA34: DLAConfig(
            pretrained_model_name="dla34.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA46_C: DLAConfig(
            pretrained_model_name="dla46_c.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA46X_C: DLAConfig(
            pretrained_model_name="dla46x_c.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA60: DLAConfig(
            pretrained_model_name="dla60.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA60X: DLAConfig(
            pretrained_model_name="dla60x.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA60X_C: DLAConfig(
            pretrained_model_name="dla60x_c.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA102: DLAConfig(
            pretrained_model_name="dla102.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA102X: DLAConfig(
            pretrained_model_name="dla102x.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA102X2: DLAConfig(
            pretrained_model_name="dla102x2.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.DLA169: DLAConfig(
            pretrained_model_name="dla169.in1k",
            source=ModelSource.TIMM,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DLA34

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None

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

        return ModelInfo(
            model="DLA",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load pretrained DLA model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DLA model instance.
        """
        model_name = self._variant_config.pretrained_model_name
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        # Cache model for use in load_inputs (to avoid reloading)
        self._cached_model = model

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for DLA model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for DLA.
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        if hasattr(self, "_cached_model") and self._cached_model is not None:
            model_for_config = self._cached_model
        else:
            model_for_config = self.load_model(dtype_override=dtype_override)

        # Use the model's own data config so crop/resize/norm are always correct
        data_config = resolve_data_config({}, model=model_for_config)
        timm_transforms = create_transform(**data_config)
        inputs = timm_transforms(image).unsqueeze(0)

        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        """Print classification results.

        Args:
            compiled_model_out: Output from the compiled model
        """
        print_compiled_model_results(compiled_model_out)
