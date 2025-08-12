# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vadv2 model loader implementation
"""
import torch
import numpy as np
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from ...tools.utils import get_file
from third_party.tt_forge_models.vadv2.pytorch.src.vad import VAD
from third_party.tt_forge_models.vadv2.pytorch.src.dataset import LiDARInstance3DBoxes


class ModelVariant(StrEnum):
    """Available Vadv2 model variants for autonomous driving."""

    VADV2_TINY = "vadv2_tiny"


class ModelLoader(ForgeModel):
    """Vadv2 model loader implementation for autonomous driving tasks."""

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VADV2_TINY

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        # Configuration parameters
        self.processor = None

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
            model="vadv2",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the Vadv2 model instance with default settings.

        Returns:
            Torch model: The Vadv2 model instance.
        """
        # Load model with defaults
        model = VAD()
        checkpoint_path = get_file("test_files/pytorch/vadv2/vadv2_tiny.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint,
            strict=False,
        )
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the Vadv2 model with default settings.

        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """
        input_dict = {
            "img_metas": [
                [
                    [
                        {
                            "ori_shape": [(360, 640, 3)] * 6,
                            "img_shape": [(384, 640, 3)] * 6,
                            "lidar2img": [
                                torch.rand(4, 4),
                                torch.rand(4, 4),
                                torch.rand(4, 4),
                                torch.rand(4, 4),
                                torch.rand(4, 4),
                                torch.rand(4, 4),
                            ],
                            "pad_shape": [(384, 640, 3)] * 6,
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
                            "can_bus": torch.tensor(
                                [
                                    6.50486842e02,
                                    1.81754303e03,
                                    0.00000000e00,
                                    1.84843146e-01,
                                    1.84843146e-01,
                                    1.84843146e-01,
                                    1.84843146e-01,
                                    8.47522666e-01,
                                    1.34135536e00,
                                    9.58588434e00,
                                    -9.57939215e-03,
                                    6.51179999e-03,
                                    3.75314295e-01,
                                    3.77446848e00,
                                    0.00000000e00,
                                    0.00000000e00,
                                    3.51370076e00,
                                    2.01320224e02,
                                ]
                            ),
                        }
                    ]
                ]
            ],
            "gt_bboxes_3d": [
                [
                    [
                        LiDARInstance3DBoxes(
                            torch.tensor(
                                [
                                    [
                                        -1.49,
                                        -14.395,
                                        -1.697,
                                        1.772,
                                        4.294,
                                        1.597,
                                        2.7079,
                                        0.033,
                                        -0.118,
                                    ],
                                    [
                                        -5.703,
                                        -16.928,
                                        -1.723,
                                        1.696,
                                        4.983,
                                        1.604,
                                        2.7118,
                                        0.04025,
                                        -0.02866,
                                    ],
                                    [
                                        2.212,
                                        -16.031,
                                        -1.756,
                                        1.764,
                                        4.135,
                                        1.565,
                                        2.7047,
                                        0.03279,
                                        -0.08034,
                                    ],
                                    [
                                        -8.515,
                                        -16.771,
                                        -1.631,
                                        1.709,
                                        4.624,
                                        1.538,
                                        2.7163,
                                        0.03411,
                                        0.05917,
                                    ],
                                    [
                                        -3.307,
                                        -19.646,
                                        -1.678,
                                        1.785,
                                        5.143,
                                        1.597,
                                        2.7124,
                                        0.03402,
                                        -0.12001,
                                    ],
                                    [
                                        -9.044,
                                        -12.747,
                                        -1.671,
                                        1.763,
                                        4.171,
                                        1.586,
                                        2.7187,
                                        0.03797,
                                        -0.11996,
                                    ],
                                    [
                                        -9.413,
                                        -9.297,
                                        -1.692,
                                        1.777,
                                        4.312,
                                        1.581,
                                        2.7201,
                                        0.04248,
                                        -0.12339,
                                    ],
                                    [
                                        -7.015,
                                        -14.134,
                                        -1.706,
                                        1.748,
                                        4.462,
                                        1.602,
                                        2.7136,
                                        0.03665,
                                        -0.09436,
                                    ],
                                    [
                                        -4.188,
                                        -16.865,
                                        -1.726,
                                        1.729,
                                        4.759,
                                        1.581,
                                        2.7098,
                                        0.03482,
                                        -0.10607,
                                    ],
                                    [
                                        -6.324,
                                        -18.912,
                                        -1.687,
                                        1.773,
                                        5.128,
                                        1.602,
                                        2.7149,
                                        0.03843,
                                        -0.11187,
                                    ],
                                    [
                                        -5.529,
                                        -15.985,
                                        -1.681,
                                        1.711,
                                        4.251,
                                        1.588,
                                        2.7102,
                                        0.03579,
                                        -0.11788,
                                    ],
                                ]
                            ),
                            box_dim=9,
                        )
                    ]
                ]
            ],
            "gt_labels_3d": [[[torch.tensor([8, 8, 8, 8, 8, 8, 0, 8, 8, 0, 8])]]],
            "fut_valid_flag": [torch.tensor([True])],
            "ego_his_trajs": [[torch.tensor([[[[0.0757, 4.2529], [0.0757, 4.2529]]]])]],
            "ego_fut_trajs": [[torch.zeros((1, 1, 6, 2))]],
            "ego_fut_masks": [[torch.ones((1, 1, 6))]],
            "ego_fut_cmd": [[torch.tensor([[[[1.0, 0.0, 0.0]]]])]],
            "ego_lcf_feat": [[torch.zeros((1, 1, 1, 9))]],
            "gt_attr_labels": [[[torch.rand(11, 34)]]],
            "map_gt_labels_3d": [[torch.zeros((7,))]],
            "map_gt_bboxes_3d": [[None]],
        }
        tensor = torch.randn(1, 6, 3, 384, 640)
        img1 = []
        img1.append(tensor)
        kwargs = {
            "img": img1,
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

        return kwargs
