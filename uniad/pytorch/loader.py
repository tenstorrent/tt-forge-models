# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UniAD model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import torch

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
from ...tools.utils import get_file


@dataclass
class UniadConfig(ModelConfig):
    """Configuration specific to UniAD models"""

    checkpoint_file: str
    img_shape: tuple = (928, 1600)  # (height, width)
    num_cameras: int = 6
    num_channels: int = 3


class ModelVariant(StrEnum):
    """Available UniAD model variants for autonomous driving."""

    UNIAD_E2E = "uniad_e2e"


class ModelLoader(ForgeModel):
    """UniAD model loader implementation for autonomous driving tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.UNIAD_E2E: UniadConfig(
            pretrained_model_name="uniad_e2e",
            checkpoint_file="test_files/pytorch/uniad/uniad_e2e.pth",
            img_shape=(928, 1600),
            num_cameras=6,
            num_channels=3,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.UNIAD_E2E

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None

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

        return ModelInfo(
            model="uniad",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the UniAD model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The UniAD model instance.
        """
        # Import model class
        from third_party.tt_forge_models.uniad.pytorch.src.uniad_e2e import UniAD

        # Get checkpoint file from the instance's variant config
        checkpoint_file = self._variant_config.checkpoint_file

        # Load model
        model = UniAD()
        checkpoint_path = get_file(checkpoint_file)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint,
            strict=False,
        )
        model.eval()

        # Store model for potential use in input preprocessing
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, **kwargs):
        """Preprocess inputs and return model-ready input dictionary.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            **kwargs: Additional keyword arguments (unused, for compatibility).

        Returns:
            dict: A dictionary of preprocessed input tensors and metadata suitable for the model.
        """
        from third_party.tt_forge_models.uniad.pytorch.src.track_utils import (
            LiDARInstance3DBoxes,
        )

        # Get configuration from variant config
        img_h, img_w = self._variant_config.img_shape
        num_cameras = self._variant_config.num_cameras
        num_channels = self._variant_config.num_channels

        # Determine dtype
        dtype = dtype_override if dtype_override is not None else torch.float32

        # Create input tensors
        img_tensor = [
            torch.randn(1, num_cameras, num_channels, img_h, img_w, dtype=dtype) * 50
        ]
        img_metas = [
            [
                {
                    "ori_shape": [(900, img_w, num_channels)] * num_cameras,
                    "img_shape": [(img_h, img_w, num_channels)] * num_cameras,
                    "lidar2img": [
                        torch.randn(4, 4, dtype=dtype) * 500
                        for _ in range(num_cameras)
                    ],
                    "box_type_3d": LiDARInstance3DBoxes,
                    "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                    "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                    "can_bus": torch.randn(18, dtype=dtype) * 300,
                }
            ]
        ]

        timestamp = [torch.tensor([1533151603.5476], dtype=dtype)]
        l2g_r_mat = torch.randn(1, 3, 3, dtype=dtype)
        l2g_t = torch.randn(1, 3, dtype=dtype) * 800

        gt_lane_labels = [torch.randint(0, 4, (1, 12))]
        gt_lane_bboxes = [torch.randint(0, 200, (1, 12, 4))]
        gt_segmentation = [torch.randint(0, 256, (1, 7, 200, 200))]
        gt_lane_masks = [torch.zeros((1, 12, 200, 200), dtype=torch.uint8)]

        gt_instance = [torch.randint(0, 318, (1, 7, 200, 200))]
        gt_centerness = [torch.randint(0, 256, (1, 7, 1, 200, 200))]
        gt_offset = [torch.randint(-5, 256, (1, 7, 2, 200, 200))]
        gt_flow = [torch.full((1, 7, 2, 200, 200), 255)]
        gt_backward_flow = [torch.full((1, 7, 2, 200, 200), 255)]
        gt_occ_has_invalid_frame = [torch.ones((1,), dtype=torch.long)]
        gt_occ_img_is_valid = [torch.randint(0, 2, (1, 9))]

        sdc_planning = [torch.rand(1, 1, 6, 3, dtype=dtype) * 25]
        sdc_planning_mask = [torch.ones(1, 1, 6, 2, dtype=dtype)]
        command = [torch.tensor([0])]

        inputs = {
            "command": command,
            "gt_backward_flow": gt_backward_flow,
            "gt_centerness": gt_centerness,
            "gt_flow": gt_flow,
            "gt_instance": gt_instance,
            "gt_lane_bboxes": gt_lane_bboxes,
            "gt_lane_labels": gt_lane_labels,
            "gt_lane_masks": gt_lane_masks,
            "gt_occ_has_invalid_frame": gt_occ_has_invalid_frame,
            "gt_occ_img_is_valid": gt_occ_img_is_valid,
            "gt_offset": gt_offset,
            "gt_segmentation": gt_segmentation,
            "img": [img_tensor],
            "img_metas": img_metas,
            "l2g_r_mat": l2g_r_mat,
            "l2g_t": l2g_t,
            "rescale": True,
            "sdc_planning": sdc_planning,
            "sdc_planning_mask": sdc_planning_mask,
            "timestamp": timestamp,
        }

        return inputs

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """
        return self.input_preprocess(dtype_override=dtype_override, **kwargs)

    def output_postprocess(self, output):
        return output
