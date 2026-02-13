# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vadv2 model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import torch
import numpy as np

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
class Vadv2Config(ModelConfig):
    """Configuration specific to Vadv2 models"""

    checkpoint_file: str
    img_shape: tuple = (384, 640)  # (height, width)
    ori_shape: tuple = (360, 640)  # Original shape before padding
    num_cameras: int = 6
    num_channels: int = 3
    num_boxes: int = 11
    box_dim: int = 9


class ModelVariant(StrEnum):
    """Available Vadv2 model variants for autonomous driving."""

    VADV2_TINY = "Tiny"


class ModelLoader(ForgeModel):
    """Vadv2 model loader implementation for autonomous driving tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.VADV2_TINY: Vadv2Config(
            pretrained_model_name="vadv2_tiny",
            checkpoint_file="test_files/pytorch/vadv2/vadv2_tiny.pth",
            img_shape=(384, 640),
            ori_shape=(360, 640),
            num_cameras=6,
            num_channels=3,
            num_boxes=11,
            box_dim=9,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VADV2_TINY

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
            model="VADv2",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Vadv2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Vadv2 model instance.
        """
        # Import model class
        from third_party.tt_forge_models.vadv2.pytorch.src.vad import VAD

        # Get checkpoint file from the instance's variant config
        checkpoint_file = self._variant_config.checkpoint_file

        # Load model
        model = VAD()
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
        from third_party.tt_forge_models.vadv2.pytorch.src.dataset import (
            LiDARInstance3DBoxes,
        )

        # Get configuration from variant config
        img_h, img_w = self._variant_config.img_shape
        ori_h, ori_w = self._variant_config.ori_shape
        num_cameras = self._variant_config.num_cameras
        num_channels = self._variant_config.num_channels
        num_boxes = self._variant_config.num_boxes
        box_dim = self._variant_config.box_dim

        # Determine dtype
        dtype = dtype_override if dtype_override is not None else torch.float32

        # Create input dictionary
        input_dict = {
            "img_metas": [
                [
                    [
                        {
                            "ori_shape": [(ori_h, ori_w, num_channels)] * num_cameras,
                            "img_shape": [(img_h, img_w, num_channels)] * num_cameras,
                            "lidar2img": [
                                torch.rand(4, 4, dtype=dtype) * 6,
                            ],
                            "pad_shape": [(img_h, img_w, num_channels)] * num_cameras,
                            "scale_factor": 1.0,
                            "flip": False,
                            "pcd_horizontal_flip": False,
                            "pcd_vertical_flip": False,
                            "box_type_3d": LiDARInstance3DBoxes,
                            "img_norm_cfg": {
                                "mean": np.array(
                                    [123.675, 116.28, 103.53], dtype=np.float32
                                ),
                                "std": np.array(
                                    [58.395, 57.12, 57.375], dtype=np.float32
                                ),
                                "to_rgb": True,
                            },
                            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                            "prev_idx": "",
                            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                            "pcd_scale_factor": 1.0,
                            "pts_filename": "data/pcd.bin",
                            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                            "can_bus": torch.rand(18, dtype=dtype),
                        }
                    ]
                ]
            ],
            "gt_bboxes_3d": [
                [
                    [
                        LiDARInstance3DBoxes(
                            torch.rand(num_boxes, box_dim, dtype=dtype),
                            box_dim=box_dim,
                        )
                    ]
                ]
            ],
            "gt_labels_3d": [[[torch.tensor([8, 8, 8, 8, 8, 8, 0, 8, 8, 0, 8])]]],
            "fut_valid_flag": [torch.tensor([True])],
            "ego_his_trajs": [
                [torch.tensor([[[[0.0757, 4.2529], [0.0757, 4.2529]]]], dtype=dtype)]
            ],
            "ego_fut_trajs": [[torch.zeros((1, 1, 6, 2), dtype=dtype)]],
            "ego_fut_masks": [[torch.ones((1, 1, 6), dtype=dtype)]],
            "ego_fut_cmd": [[torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=dtype)]],
            "ego_lcf_feat": [[torch.zeros((1, 1, 1, 9), dtype=dtype)]],
            "gt_attr_labels": [[[torch.rand(num_boxes, 34, dtype=dtype)]]],
            "map_gt_labels_3d": [[torch.zeros((7,), dtype=dtype)]],
            "map_gt_bboxes_3d": [[None]],
        }

        # Create image tensor
        tensor = torch.randn(1, num_cameras, num_channels, img_h, img_w, dtype=dtype)
        img = [tensor]

        # Build inputs dictionary
        inputs = {
            "img": img,
            "img_metas": input_dict["img_metas"],
            "gt_bboxes_3d": input_dict["gt_bboxes_3d"],
            "gt_labels_3d": input_dict["gt_labels_3d"],
            "fut_valid_flag": input_dict["fut_valid_flag"],
            "ego_his_trajs": input_dict["ego_his_trajs"],
            "ego_fut_trajs": input_dict["ego_fut_trajs"],
            "ego_fut_cmd": input_dict["ego_fut_cmd"],
            "ego_lcf_feat": input_dict["ego_lcf_feat"],
            "gt_attr_labels": input_dict["gt_attr_labels"],
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
