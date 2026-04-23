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
        Create synthetic inputs for action sampling.
        Returns images, image masks, language tokens, language masks, state,
        and a pre-generated noise tensor for deterministic diffusion sampling.
        """
        import torch

        config = self.pi_05.config
        batch_size = 1
        h, w = config.image_resolution

        images = []
        img_masks = []
        for _ in config.image_features:
            images.append(torch.zeros(batch_size, 3, h, w))
            img_masks.append(torch.ones(batch_size, dtype=torch.bool))
        for _ in range(config.empty_cameras):
            images.append(torch.full((batch_size, 3, h, w), -1.0))
            img_masks.append(torch.zeros(batch_size, dtype=torch.bool))

        lang_tokens = torch.zeros(
            batch_size, config.tokenizer_max_length, dtype=torch.long
        )
        lang_masks = torch.ones(
            batch_size, config.tokenizer_max_length, dtype=torch.bool
        )
        state = torch.zeros(batch_size, config.max_state_dim)
        noise = self.pi_05.model.sample_noise(
            (batch_size, config.chunk_size, config.max_action_dim),
            device=state.device,
        )

        return images, img_masks, lang_tokens, lang_masks, state, noise

    def postprocess(self, pred_action):
        """Apply postprocessing to predicted actions."""
        return self.postprocess_fn(pred_action)
