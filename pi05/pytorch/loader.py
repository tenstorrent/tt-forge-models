# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi-0.5 (pi05) model loader implementation for action prediction tasks
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
    """pi05 config with an optional pinned HF Hub revision."""

    revision: Optional[str] = None


class ModelVariant(StrEnum):
    """Available pi05 model variants."""

    BASE = "pi05_base"


class ModelLoader(ForgeModel):
    """pi05 model loader implementation for action prediction task."""

    # Dictionary of available model variants
    _VARIANTS = {
        # Pin to the last revision compatible with lerobot==0.4.3; the newer
        # revision (#6, "Add relative action processor steps") adds a
        # `relative_actions_processor` step the pinned library can't load.
        ModelVariant.BASE: Pi05ModelConfig(
            pretrained_model_name="lerobot/pi05_base",
            revision="a538eb273274eb30f126a118f39dbc0ee212c883",
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
            model="pi05",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the pi05 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to cast the model weights to.

        Returns:
            torch.nn.Module: The pi05 Policy instance.
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

        self.pi05 = get_custom_pi05_policy(self.pretrained_model_name)

        if dtype_override is not None:
            self.pi05 = self.pi05.to(dtype_override)

        self.pi05.eval()
        return self.pi05

    def load_inputs(self, dtype_override=None, episode_index=0):
        """
        Load and preprocess inputs for action sampling.

        Returns images, image masks, language tokens, language masks, and a
        pre-generated noise tensor for deterministic flow-matching sampling.
        Unlike pi0, pi05 folds the proprioceptive state into the language
        prompt, so no separate state tensor is returned.
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.preprocess, self.postprocess_fn = make_pre_post_processors(
            self.pi05.config,
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
        ) = self.pi05.preprocess_for_sampling(batch)

        bsize = lang_tokens.shape[0]
        noise = self.pi05.model.sample_noise(
            (bsize, self.pi05.config.chunk_size, self.pi05.config.max_action_dim),
            device=lang_tokens.device,
        )

        if dtype_override is not None:
            images = [img.to(dtype_override) for img in images]
            noise = noise.to(dtype_override)

        return images, img_masks, lang_tokens, lang_masks, noise

    def postprocess(self, pred_action):
        """Apply postprocessing to predicted actions."""
        return self.postprocess_fn(pred_action)
