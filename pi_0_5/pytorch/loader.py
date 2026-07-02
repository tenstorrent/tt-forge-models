# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi-0.5 model loader implementation for action prediction tasks
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
        # Pin to the last revision before "Add relative action processor steps"
        # (commit 7de6639, 2026-06-03). That newer revision's saved
        # policy_preprocessor.json references a `relative_actions_processor`
        # step that lerobot==0.4.3 does not register, so make_pre_post_processors
        # can't load it. This revision keeps the identical weights with a
        # 0.4.3-loadable processor pipeline.
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
        self.pi_0_5 = None
        self.preprocess = None
        self.postprocess_fn = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="pi_0_5",
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

        self.pi_0_5 = get_custom_pi05_policy(self.pretrained_model_name)
        self.pi_0_5.eval()

        if dtype_override is not None:
            self.pi_0_5 = self.pi_0_5.to(dtype_override)

        return self.pi_0_5

    def load_inputs(self, dtype_override=None, **kwargs):
        """
        Load and preprocess inputs for action sampling.

        Pi-0.5 folds the proprioceptive state into the language prompt during
        preprocessing (unlike Pi-0, which passes a separate ``state`` tensor to
        the model), so the model inputs are images + language tokens only.

        A small, deterministic synthetic observation (three camera frames, a
        32-d state vector and a task string) is built and run through the model's
        saved pre-processor so no external dataset download is required. A
        pre-generated noise tensor is returned so both CPU and device runs use
        the same starting noise for a reproducible flow-matching comparison.

        Returns:
            images, img_masks, lang_tokens, lang_masks, noise
        """
        from lerobot.policies.factory import make_pre_post_processors

        if self.pi_0_5 is None:
            self.load_model(dtype_override=dtype_override)

        self.preprocess, self.postprocess_fn = make_pre_post_processors(
            self.pi_0_5.config,
            self.pretrained_model_name,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        # Deterministic synthetic single-frame observation matching the model's
        # declared input_features (3 cameras @ 3x224x224 in [0, 1], 32-d state).
        gen = torch.Generator().manual_seed(0)
        frame = {
            "observation.images.base_0_rgb": torch.rand(3, 224, 224, generator=gen),
            "observation.images.left_wrist_0_rgb": torch.rand(
                3, 224, 224, generator=gen
            ),
            "observation.images.right_wrist_0_rgb": torch.rand(
                3, 224, 224, generator=gen
            ),
            "observation.state": torch.rand(32, generator=gen),
            "task": "pick up the cube and place it on the plate",
        }

        batch = self.preprocess(frame)
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        ) = self.pi_0_5.preprocess_for_sampling(batch)

        bsize = lang_tokens.shape[0]
        noise = self.pi_0_5.model.sample_noise(
            (bsize, self.pi_0_5.config.chunk_size, self.pi_0_5.config.max_action_dim),
            device=lang_tokens.device,
        )

        if dtype_override is not None:
            images = [img.to(dtype_override) for img in images]
            noise = noise.to(dtype_override)

        return images, img_masks, lang_tokens, lang_masks, noise

    def postprocess(self, pred_action):
        """Apply postprocessing to predicted actions."""
        return self.postprocess_fn(pred_action)
