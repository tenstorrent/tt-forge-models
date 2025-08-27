# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Detr3d model loader implementation
"""
import torch
import numpy as np
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from ...tools.utils import get_file
from third_party.tt_forge_models.detr3d.pytorch.src.load_model import load_model
from third_party.tt_forge_models.detr3d.pytorch.src.projects.mmdet3d_plugin.datasets.lidar_box3d import (
    LiDARInstance3DBoxes,
)


class ModelVariant(StrEnum):
    """Available Detr3d model variants for autonomous driving."""

    DETR3D_RESNET101 = "detr3d_resnet101"


class ModelLoader(ForgeModel):
    """Detr3d model loader implementation for autonomous driving tasks."""

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DETR3D_RESNET101

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
            model="detr3d",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the Detr3d model instance with default settings.
        Returns:
            Torch model: The Detr3d model instance.
        """
        # Load model with defaults
        model = load_model()
        checkpoint_path = get_file("test_files/pytorch/detr3d/detr3d_resnet101.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint,
            strict=False,
        )
        model = model.eval()
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the Detr3d model with default settings.
        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """
        input_dict = {
            "img_metas": [
                [
                    [
                        {
                            "filename": [
                                get_file(
                                    "test_images/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
                                ),
                                get_file(
                                    "test_images/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg"
                                ),
                                get_file(
                                    "test_images/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg"
                                ),
                                get_file(
                                    "test_images/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg"
                                ),
                                get_file(
                                    "test_images/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg"
                                ),
                                get_file(
                                    "test_images/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg"
                                ),
                            ],
                            "ori_shape": [(900, 1600, 3, 6)],
                            "img_shape": [(928, 1600, 3)] * 6,
                            "lidar2img": [
                                np.array(
                                    [
                                        [
                                            1.24298977e03,
                                            8.40649523e02,
                                            3.27625534e01,
                                            -3.54351139e02,
                                        ],
                                        [
                                            -1.82012609e01,
                                            5.36798564e02,
                                            -1.22553754e03,
                                            -6.44707879e02,
                                        ],
                                        [
                                            -1.17025046e-02,
                                            9.98471159e-01,
                                            5.40221896e-02,
                                            -4.25203639e-01,
                                        ],
                                        [
                                            0.00000000e00,
                                            0.00000000e00,
                                            0.00000000e00,
                                            1.00000000e00,
                                        ],
                                    ]
                                ),
                                np.array(
                                    [
                                        [
                                            1.36494654e03,
                                            -6.19264860e02,
                                            -4.03391641e01,
                                            -4.61642859e02,
                                        ],
                                        [
                                            3.79462336e02,
                                            3.20307276e02,
                                            -1.23979473e03,
                                            -6.92556280e02,
                                        ],
                                        [
                                            8.43406855e-01,
                                            5.36312055e-01,
                                            3.21598489e-02,
                                            -6.10371854e-01,
                                        ],
                                        [
                                            0.00000000e00,
                                            0.00000000e00,
                                            0.00000000e00,
                                            1.00000000e00,
                                        ],
                                    ]
                                ),
                                np.array(
                                    [
                                        [
                                            3.23698342e01,
                                            1.50315427e03,
                                            7.76231827e01,
                                            -3.02437885e02,
                                        ],
                                        [
                                            -3.89320197e02,
                                            3.20441551e02,
                                            -1.23745300e03,
                                            -6.79424755e02,
                                        ],
                                        [
                                            -8.23415292e-01,
                                            5.65940098e-01,
                                            4.12196894e-02,
                                            -5.29677094e-01,
                                        ],
                                        [
                                            0.00000000e00,
                                            0.00000000e00,
                                            0.00000000e00,
                                            1.00000000e00,
                                        ],
                                    ]
                                ),
                                np.array(
                                    [
                                        [
                                            -8.03982245e02,
                                            -8.50723862e02,
                                            -2.64376631e01,
                                            -8.70795988e02,
                                        ],
                                        [
                                            1.08232816e01,
                                            -4.45285963e02,
                                            -8.14897443e02,
                                            -7.08684241e02,
                                        ],
                                        [
                                            -8.33350064e-03,
                                            -9.99200442e-01,
                                            -3.91028008e-02,
                                            -1.01645350e00,
                                        ],
                                        [
                                            0.00000000e00,
                                            0.00000000e00,
                                            0.00000000e00,
                                            1.00000000e00,
                                        ],
                                    ]
                                ),
                                np.array(
                                    [
                                        [
                                            -1.18656611e03,
                                            9.23261441e02,
                                            5.32641592e01,
                                            -6.25341190e02,
                                        ],
                                        [
                                            -4.62625515e02,
                                            -1.02540587e02,
                                            -1.25247717e03,
                                            -5.61828455e02,
                                        ],
                                        [
                                            -9.47586752e-01,
                                            -3.19482867e-01,
                                            3.16948959e-03,
                                            -4.32527296e-01,
                                        ],
                                        [
                                            0.00000000e00,
                                            0.00000000e00,
                                            0.00000000e00,
                                            1.00000000e00,
                                        ],
                                    ]
                                ),
                                np.array(
                                    [
                                        [
                                            2.85189233e02,
                                            -1.46927652e03,
                                            -5.95634293e01,
                                            -2.72600319e02,
                                        ],
                                        [
                                            4.44736043e02,
                                            -1.22825702e02,
                                            -1.25039267e03,
                                            -5.88246117e022,
                                        ],
                                        [
                                            9.24052925e-01,
                                            -3.82246554e-01,
                                            -3.70989150e-03,
                                            -4.64645142e-01,
                                        ],
                                        [
                                            0.00000000e00,
                                            0.00000000e00,
                                            0.00000000e00,
                                            1.00000000e00,
                                        ],
                                    ]
                                ),
                            ],
                            "pad_shape": [(928, 1600, 3)] * 6,
                            "scale_factor": 1.0,
                            "flip": False,
                            "pcd_horizontal_flip": False,
                            "pcd_vertical_flip": False,
                            "box_type_3d": LiDARInstance3DBoxes,
                            "img_norm_cfg": {
                                "mean": np.array(
                                    [103.53, 116.28, 123.675], dtype=np.float32
                                ),
                                "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                                "to_rgb": False,
                            },
                            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                            "prev_idx": "",
                            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                            "pcd_scale_factor": 1.0,
                            "pts_filename": "data/pcd.bin",
                            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                        }
                    ]
                ]
            ],
        }
        tensor = torch.randn(1, 6, 3, 928, 1600)
        img = []
        img.append(tensor)
        kwargs = {
            "img": img,
            "img_metas": input_dict["img_metas"],
        }

        return kwargs
