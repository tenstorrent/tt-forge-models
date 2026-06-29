# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi-0.5 (Pi05) model loader implementation for action prediction tasks
"""
import torch
from dataclasses import dataclass
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


@dataclass
class Pi05ModelConfig(ModelConfig):
    """Pi-0.5 config with an optional pinned HF Hub revision."""

    revision: Optional[str] = None


class ModelVariant(StrEnum):
    """Available Pi-0.5 model variants."""

    BASE = "pi05_base"


class ModelLoader(ForgeModel):
    """Pi-0.5 model loader implementation for action prediction task."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: Pi05ModelConfig(
            pretrained_model_name="lerobot/pi05_base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

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
            variant: Optional variant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="pi_05",
            variant=variant,
            group=ModelGroup.RED,
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

        # When a revision is pinned, snapshot the repo locally and load weights
        # and processor config from that path (make_pre_post_processors can't
        # take a revision directly).
        revision = getattr(self._variant_config, "revision", None)
        if revision is not None:
            from huggingface_hub import snapshot_download

            self.pretrained_model_name = snapshot_download(
                repo_id=self._variant_config.pretrained_model_name,
                revision=revision,
            )
        else:
            self.pretrained_model_name = self._variant_config.pretrained_model_name

        self.pi_05 = get_custom_pi05_policy(self.pretrained_model_name)
        self.pi_05.eval()
        return self.pi_05

    def load_inputs(self, dtype_override=None, episode_index=0):
        """
        Load and preprocess inputs for action sampling.

        Returns images, image masks, language tokens, language masks, and a
        pre-generated noise tensor for deterministic flow-matching sampling.
        Pi-0.5 does not take a separate state tensor -- the proprioceptive
        state is discretized into the language prompt by the pre-processor.
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # The pi05_base processor configs reference the "relative_actions_processor" /
        # "absolute_actions_processor" steps, which were added to lerobot after the pinned
        # 0.4.3 release. Importing this module registers compatible implementations so the
        # processors load. (Both are enabled=false for this checkpoint -> pass-through.)
        from .src import relative_action_processor  # noqa: F401

        self.preprocess, self.postprocess_fn = make_pre_post_processors(
            self.pi_05.config,
            self.pretrained_model_name,
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
        ) = self.pi_05.preprocess_for_sampling(batch)

        bsize = lang_tokens.shape[0]
        noise = self.pi_05.model.sample_noise(
            (bsize, self.pi_05.config.chunk_size, self.pi_05.config.max_action_dim),
            device=lang_tokens.device,
        )

        return images, img_masks, lang_tokens, lang_masks, noise

    def postprocess(self, pred_action):
        """Apply postprocessing to predicted actions."""
        return self.postprocess_fn(pred_action)
