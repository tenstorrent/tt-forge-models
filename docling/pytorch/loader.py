# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Docling document layout analysis model loader implementation for object detection.
"""
import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Docling layout model variants for document layout analysis."""

    HERON = "Heron"
    EGRET_LARGE = "Egret_Large"
    EGRET_XLARGE = "Egret_XLarge"


class ModelLoader(ForgeModel):
    """Docling layout model loader for document layout analysis tasks."""

    _VARIANTS = {
        ModelVariant.HERON: ModelConfig(
            pretrained_model_name="docling-project/docling-layout-heron",
        ),
        ModelVariant.EGRET_LARGE: ModelConfig(
            pretrained_model_name="docling-project/docling-layout-egret-large",
        ),
        ModelVariant.EGRET_XLARGE: ModelConfig(
            pretrained_model_name="docling-project/docling-layout-egret-xlarge",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HERON

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Docling_Layout",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = RTDetrImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def _load_model_class(self):
        """Return the appropriate model class for the current variant."""
        if self._variant in (ModelVariant.EGRET_LARGE, ModelVariant.EGRET_XLARGE):
            from transformers import DFineForObjectDetection

            return DFineForObjectDetection
        return RTDetrV2ForObjectDetection

    @staticmethod
    def _patch_dfine_compilable_check():
        # DFine's torch_compilable_check asserts spatial shapes match sequence_length.
        # Under torch.compile this check is a compile-time assertion (no-op at runtime).
        # Under TorchXLA it runs on-device, but TT hardware computes int64 .sum() in
        # bfloat16 (8400 rounds to 8384), producing a false-positive failure.
        # The condition is guaranteed true by construction in DFineModel.forward —
        # spatial_shapes and source_flatten are built from the same sources in the
        # same loop. Making the check a no-op for non-CPU tensors matches torch.compile
        # behavior and is safe here.
        import transformers.models.d_fine.modeling_d_fine as _dfine_module
        from transformers.utils.import_utils import (
            torch_compilable_check as _orig_check,
        )

        def _xla_aware_compilable_check(cond, msg, error_type=ValueError):
            if isinstance(cond, torch.Tensor) and cond.device.type != "cpu":
                return
            return _orig_check(cond, msg, error_type)

        _dfine_module.torch_compilable_check = _xla_aware_compilable_check

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model_class = self._load_model_class()
        if model_class.__name__ == "DFineForObjectDetection":
            self._patch_dfine_compilable_check()
        model = model_class.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
