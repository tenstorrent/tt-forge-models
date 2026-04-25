# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isaac GR00T model loader implementation for robotic policy inference.
"""

from typing import Optional, Dict, Any
import torch
import numpy as np

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
    """Available Isaac GR00T model variants."""

    GROOT_N1_5_3B = "Gr00t_N1.5_3B"
    GROOT_N1_6_3B = "Gr00t_N1.6_3B"
    GROOT_N1_6_DROID = "Gr00t_N1.6_DROID"


class ModelLoader(ForgeModel):
    """Isaac GR00T model loader for robotic policy inference."""

    _VARIANTS = {
        ModelVariant.GROOT_N1_5_3B: ModelConfig(
            pretrained_model_name="nvidia/GR00T-N1.5-3B",
        ),
        ModelVariant.GROOT_N1_6_3B: ModelConfig(
            pretrained_model_name="nvidia/GR00T-N1.6-3B",
        ),
        ModelVariant.GROOT_N1_6_DROID: ModelConfig(
            pretrained_model_name="nvidia/GR00T-N1.6-DROID",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GROOT_N1_5_3B
    DEFAULT_DATA_CONFIG = "fourier_gr1_arms_only"
    DEFAULT_EMBODIMENT = "gr1"
    DEFAULT_DENOISING_STEPS = 4

    # Per-variant overrides for variants fine-tuned on a non-GR1 embodiment.
    _VARIANT_DEFAULTS = {}

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        data_config: Optional[str] = None,
        embodiment_tag: Optional[str] = None,
        denoising_steps: Optional[int] = None,
    ):
        """
        Initialize Isaac GR00T model loader.

        Args:
            variant: Model variant to load (default: GROOT_N1_5_3B)
            data_config: Data configuration name (default: fourier_gr1_arms_only)
            embodiment_tag: Embodiment tag (default: gr1)
            denoising_steps: Number of denoising steps (default: 4)
        """
        super().__init__(variant)
        variant_defaults = self._VARIANT_DEFAULTS.get(self._variant, {})
        self.data_config = (
            data_config
            or variant_defaults.get("data_config")
            or self.DEFAULT_DATA_CONFIG
        )
        self.embodiment_tag = (
            embodiment_tag
            or variant_defaults.get("embodiment_tag")
            or self.DEFAULT_EMBODIMENT
        )
        self.denoising_steps = denoising_steps or self.DEFAULT_DENOISING_STEPS

        # Lazy initialization
        self._modality_config = None
        self._modality_transform = None
        self._model = None

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
        group = (
            ModelGroup.VULCAN
            if variant in (ModelVariant.GROOT_N1_6_3B, ModelVariant.GROOT_N1_6_DROID)
            else ModelGroup.RED
        )
        return ModelInfo(
            model="ISAAC-GR00T",
            variant=variant,
            group=group,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def _load_data_config(self):
        """Load data configuration."""
        from .src.utils import load_data_config

        if self._modality_config is None:
            data_config_obj = load_data_config(self.data_config)
            self._modality_config = data_config_obj.modality_config()
            self._modality_transform = data_config_obj.transform()

    def load_model(self, *, dtype_override=None, **kwargs):
        """
        Load and return the Isaac GR00T policy model.

        Args:
            dtype_override: Optional dtype to cast model to (not recommended for GR00T)

        Returns:
            Gr00tPolicyModule instance (nn.Module)
        """
        from .src.model import Gr00tPolicyModule

        self._load_data_config()

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Create policy module
        policy = Gr00tPolicyModule(
            model_path=pretrained_model_name,
            embodiment_tag=self.embodiment_tag,
            modality_config=self._modality_config,
            modality_transform=self._modality_transform,
            denoising_steps=self.denoising_steps,
        )
        policy.eval()

        if dtype_override:
            print(
                f"Warning: GR00T uses mixed precision (BFloat16/Float32). dtype_override may cause issues."
            )

        # Store model reference for preprocessing in load_inputs
        self._model = policy

        return policy

    def load_inputs(self, dtype_override=None):
        """
        Load and return synthetic preprocessed inputs for Isaac GR00T.

        Generates synthetic inputs using the Eagle processor on a dummy image,
        bypassing the LeRobotSingleDataset which requires IRD_LF_CACHE.

        Args:
            dtype_override: Optional dtype to cast inputs to (not recommended for GR00T)

        Returns:
            Dictionary containing preprocessed observation tensors ready for model inference
        """
        from PIL import Image
        from .src.model import build_eagle_processor, collate
        from .src.utils import EMBODIMENT_TAG_MAPPING, EmbodimentTag

        # Ensure model is loaded
        if not hasattr(self, "_model") or self._model is None:
            raise RuntimeError(
                "Model must be loaded before loading inputs. "
                "Call load_model() first."
            )

        self._load_data_config()

        # Build Eagle processor from cached files (no IRD_LF_CACHE needed)
        eagle_processor = build_eagle_processor()

        # Count video views from modality config
        n_views = len(self._modality_config["video"].modality_keys)

        # Create dummy PIL images (224x224 black, one per camera view)
        dummy_images = [
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            for _ in range(n_views)
        ]

        # Build eagle conversation identical to GR00TTransform._apply_vlm_processing
        instruction = "Perform the default behavior."
        eagle_conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in dummy_images],
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        text_list = [
            eagle_processor.py_apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = eagle_processor.process_vision_info(
            eagle_conversation
        )

        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }

        # Get state dimensions from model config
        model_config = self._model.policy.model.config
        max_state_dim = getattr(model_config, "max_state_dim", 128)
        state_horizon = 1

        # Embodiment ID lookup
        try:
            embodiment_id = EMBODIMENT_TAG_MAPPING[
                EmbodimentTag(self.embodiment_tag).value
            ]
        except (KeyError, ValueError):
            embodiment_id = 0

        features = [
            {
                "eagle_content": eagle_content,
                "state": np.zeros((state_horizon, max_state_dim), dtype=np.float32),
                "state_mask": np.zeros((state_horizon, max_state_dim), dtype=bool),
                "embodiment_id": embodiment_id,
            }
        ]

        inputs = collate(features, eagle_processor)
        inputs["_was_batched"] = True
        return inputs

    def postprocess(self, raw_action_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Postprocess raw action tensor to action dictionary.

        Args:
            raw_action_tensor: Raw action tensor from model forward (with return_raw=True)

        Returns:
            Dictionary containing unnormalized actions (e.g., 'action.left_arm', 'action.right_arm', etc.)
        """
        if self._model is None:
            raise RuntimeError(
                "Model must be loaded before postprocessing. Call load_model() first."
            )

        # Always use is_batch=True since we always work with batched inputs
        return self._model.postprocess(raw_action_tensor, is_batch=True)
