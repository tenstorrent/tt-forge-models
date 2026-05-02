# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MM Grounding DINO model loader implementation for zero-shot object detection.
"""
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from typing import Optional


def _patch_mm_grounding_dino_dtype():
    """Patch MM Grounding DINO model to fix dtype mismatches when running with bfloat16.

    Two locations always produce float32 tensors regardless of model dtype:
    1. get_text_position_embeddings / get_sine_pos_embed: text position embeddings are
       always float32. Added to bfloat16 text_features in with_pos_embed(), promoting
       queries to float32. The subsequent F.linear fails (float32 vs bfloat16 weight).
    2. get_reference_points: reference points are always float32. Adding float32
       reference_points to bfloat16 sampling_offsets promotes sampling_locations to
       float32. grid_sample then fails (bfloat16 value vs float32 grid).
    """
    from transformers.models.mm_grounding_dino.modeling_mm_grounding_dino import (
        MMGroundingDinoEncoderLayer,
        MultiScaleDeformableAttention,
    )

    _orig_get_text_pos = MMGroundingDinoEncoderLayer.get_text_position_embeddings

    def _patched_get_text_pos(self, text_features, text_position_embedding, text_position_ids):
        result = _orig_get_text_pos(self, text_features, text_position_embedding, text_position_ids)
        if result is not None:
            result = result.to(text_features.dtype)
        return result

    MMGroundingDinoEncoderLayer.get_text_position_embeddings = _patched_get_text_pos

    import torch.nn as nn

    _orig_msda_forward = MultiScaleDeformableAttention.forward

    def _patched_msda_forward(
        self,
        value,
        value_spatial_shapes,
        value_spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        sampling_locations = sampling_locations.to(value.dtype)
        return _orig_msda_forward(
            self,
            value,
            value_spatial_shapes,
            value_spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )

    MultiScaleDeformableAttention.forward = _patched_msda_forward


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
    """Available MM Grounding DINO model variants for zero-shot object detection."""

    BASE_ALL = "Base_All"
    LARGE_ALL = "Large_All"
    TINY_O365V1_GOLDG_V3DET = "Tiny_O365V1_GoldG_V3Det"


class ModelLoader(ForgeModel):
    """MM Grounding DINO model loader implementation for zero-shot object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_ALL: ModelConfig(
            pretrained_model_name="openmmlab-community/mm_grounding_dino_base_all",
        ),
        ModelVariant.LARGE_ALL: ModelConfig(
            pretrained_model_name="openmmlab-community/mm_grounding_dino_large_all",
        ),
        ModelVariant.TINY_O365V1_GOLDG_V3DET: ModelConfig(
            pretrained_model_name="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE_ALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.image = None
        self.text_labels = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="MM-Grounding-DINO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_ZS_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MM Grounding DINO model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MM Grounding DINO model instance for zero-shot object detection.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if dtype_override is not None:
            _patch_mm_grounding_dino_dtype()

        model_kwargs = {"return_dict": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the MM Grounding DINO model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        self.image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        self.text_labels = [["a cat", "a remote control"]]

        inputs = self.processor(
            images=self.image, text=self.text_labels, return_tensors="pt"
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
