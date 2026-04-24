# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi-0.5 model loader implementation for action prediction tasks
"""
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
    """Available Pi-0.5 model variants."""

    LIBERO_BASE = "lerobot_pi05_libero_base"
    FLOW_V2_FEB24 = "rayhanfahmed_pi05_flow_v2_feb24"


class ModelLoader(ForgeModel):
    """Pi-0.5 model loader implementation for action prediction task."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LIBERO_BASE: ModelConfig(
            pretrained_model_name="lerobot/pi05_libero_base",
        ),
        ModelVariant.FLOW_V2_FEB24: ModelConfig(
            pretrained_model_name="rayhanfahmed/pi05-flow-v2-feb24",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LIBERO_BASE

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="pi_05",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Pi-0.5 model instance with default settings.

        Returns:
            torch.nn.Module: The Pi-0.5 Policy instance.
        """

        from .src.model import get_custom_pi05_policy

        self.pretrained_model_name = self._variant_config.pretrained_model_name
        self.pi_05 = get_custom_pi05_policy(self.pretrained_model_name)
        self.pi_05.eval()
        return self.pi_05

    def load_inputs(self, dtype_override=None, episode_index=0):
        """
        Load and preprocess inputs for action sampling.
        Returns images, image masks, language tokens, language masks, state,
        and a pre-generated noise tensor for deterministic diffusion sampling.
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        try:
            self.preprocess, self.postprocess_fn = make_pre_post_processors(
                self.pi_05.config,
                self.pretrained_model_name,
                preprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
        except FileNotFoundError:
            # Fall back to creating processors from config if policy_preprocessor.json is missing
            self.preprocess, self.postprocess_fn = make_pre_post_processors(
                self.pi_05.config,
                preprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
        dataset = LeRobotDataset("lerobot/libero")
        frame_index = dataset.meta.episodes["dataset_from_index"][episode_index]
        frame = dict(dataset[frame_index])
        batch = self.preprocess(frame)
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self.pi_05.preprocess_for_sampling(batch)

        bsize = state.shape[0]
        noise = self.pi_05.model.sample_noise(
            (bsize, self.pi_05.config.chunk_size, self.pi_05.config.max_action_dim),
            device=state.device,
        )

        return images, img_masks, lang_tokens, lang_masks, state, noise

    def postprocess(self, pred_action):
        """Apply postprocessing to predicted actions."""
        return self.postprocess_fn(pred_action)
