# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MM Grounding DINO model loader implementation for zero-shot object detection.
"""
import requests
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import transformers.models.mm_grounding_dino.modeling_mm_grounding_dino as _mm_gdino_mod
from transformers.models.mm_grounding_dino.modeling_mm_grounding_dino import (
    MMGroundingDinoEncoder,
    MMGroundingDinoEncoderLayer,
    MultiScaleDeformableAttention,
)
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

        model_kwargs = {"return_dict": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if dtype_override is not None:
            # transformers hard-codes dtype=torch.float32 in several places (position
            # embeddings, reference points, get_sine_pos_embed) causing Float/BFloat16
            # mismatches when the model is loaded in bfloat16. Patch these to preserve
            # the surrounding tensor dtype instead.

            # 1. Text position embeddings (encoder)
            _orig_get_pos = MMGroundingDinoEncoderLayer.get_text_position_embeddings

            def _typed_get_pos(
                self, text_features, text_position_embedding, text_position_ids
            ):
                result = _orig_get_pos(
                    self, text_features, text_position_embedding, text_position_ids
                )
                return result.to(text_features.dtype) if result is not None else result

            MMGroundingDinoEncoderLayer.get_text_position_embeddings = _typed_get_pos

            # 2. Multi-scale deformable attention (float32 sampling grids vs bf16 values)
            _orig_msda_fwd = MultiScaleDeformableAttention.forward

            def _typed_msda_fwd(self, value, *args, **kwargs):
                new_args = [
                    a.to(value.dtype)
                    if isinstance(a, torch.Tensor) and a.is_floating_point()
                    else a
                    for a in args
                ]
                new_kwargs = {
                    k: v.to(value.dtype)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v
                    for k, v in kwargs.items()
                }
                return _orig_msda_fwd(self, value, *new_args, **new_kwargs)

            MultiScaleDeformableAttention.forward = _typed_msda_fwd

            # 3. get_sine_pos_embed creates float32 dim_t regardless of input dtype;
            #    cast output back to the input pos_tensor's dtype.
            _orig_sine = _mm_gdino_mod.get_sine_pos_embed

            def _typed_sine(
                pos_tensor, num_pos_feats=128, temperature=10000, exchange_xy=True
            ):
                result = _orig_sine(pos_tensor, num_pos_feats, temperature, exchange_xy)
                return result.to(pos_tensor.dtype)

            _mm_gdino_mod.get_sine_pos_embed = _typed_sine

            # 4. get_reference_points uses hard-coded float32 linspace; make it follow
            #    valid_ratios.dtype so downstream tensors stay in the model dtype.
            _orig_ref_pts = MMGroundingDinoEncoder.get_reference_points.__func__

            @staticmethod
            def _typed_ref_pts(spatial_shapes_list, valid_ratios, device):
                result = _orig_ref_pts(spatial_shapes_list, valid_ratios, device)
                return result.to(valid_ratios.dtype)

            MMGroundingDinoEncoder.get_reference_points = _typed_ref_pts

        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if dtype_override is not None:
            model = model.to(dtype_override)

            # 5. MMGroundingDinoModel.forward forces valid_ratios to float32 before
            #    passing to encoder/decoder. Register hooks to cast it back to the
            #    model dtype so that reference_points and downstream tensors stay bf16.
            def _cast_valid_ratios_to_bf16(module, args, kwargs):
                if "valid_ratios" in kwargs and kwargs["valid_ratios"] is not None:
                    kwargs["valid_ratios"] = kwargs["valid_ratios"].to(dtype_override)
                return args, kwargs

            model.model.encoder.register_forward_pre_hook(
                _cast_valid_ratios_to_bf16, with_kwargs=True
            )
            model.model.decoder.register_forward_pre_hook(
                _cast_valid_ratios_to_bf16, with_kwargs=True
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

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

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
