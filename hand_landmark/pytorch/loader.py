# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hand Landmark model loader implementation
"""
import os
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """Hand Landmark model loader implementation."""

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Hand Landmark model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            callable: The Hand Landmark detector function.
        """
        # Download model file
        model_asset_path = Path(__file__).parent / "hand_landmarker.task"
        if not model_asset_path.exists():
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            os.system(f"wget -P {Path(__file__).parent} -nc {url}")

        base_options = python.BaseOptions(model_asset_path=str(model_asset_path))
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)
        return detector.detect

    @classmethod
    def load_inputs(cls, batch_size=1):
        """Load and return sample inputs for the Hand Landmark model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            list: List of input images for the model.
        """
        image_file = get_file(
            "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg"
        )
        image = mp.Image.create_from_file(str(image_file))

        # Replicate inputs for batch size
        return [image] * batch_size
