# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
rayhanfahmed/eric-hf-train-5000 loader.

This is a Pi-0.5 (PI05) policy fine-tuned from ``lerobot/pi05_base`` on a
custom ``droyd_lerobot_dataset_2`` (7-dim state, 7-dim action, two camera
views renamed to the standard PI05 keys by the bundled policy preprocessor).
The loader reuses the shared PI05 inference wrappers from the ``pi_05``
package and drives the model with synthetic tensors shaped to match the
published ``config.json`` — the training dataset is not publicly available,
so using it would not work outside the original training environment.
"""
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available eric-hf-train-5000 variants."""

    DEFAULT = "rayhanfahmed/eric-hf-train-5000"


class ModelLoader(ForgeModel):
    """Loader for the rayhanfahmed/eric-hf-train-5000 PI05 policy fine-tune."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="rayhanfahmed/eric-hf-train-5000",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional StrEnum specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="eric_hf_train_5000",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the fine-tuned PI05 policy.

        Returns:
            torch.nn.Module: The PI05Policy instance with the custom
                ``forward``/``preprocess_for_sampling`` overrides supplied by
                the shared ``pi_05`` package.
        """
        from ...pi_05.pytorch.src.model import get_custom_pi05_policy

        self.pretrained_model_name = self._variant_config.pretrained_model_name
        self.policy = get_custom_pi05_policy(self.pretrained_model_name)
        self.policy.eval()
        return self.policy

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build synthetic inputs matching the published model configuration.

        The training dataset (``droyd_lerobot_dataset_2``) is not published,
        so we construct tensors that match the shapes produced by the PI05
        preprocessor: one image/mask per visual input feature, tokenised
        language filled to ``tokenizer_max_length`` and a padded state of
        ``max_state_dim``.

        Returns:
            Tuple: ``(images, img_masks, lang_tokens, lang_masks, state, noise)``
            where ``images`` and ``img_masks`` are lists of per-camera tensors
            (one entry per camera), ready to be fed to ``self.policy.forward``.
        """
        config = self.policy.config
        num_cameras = len(config.image_features)
        height, width = config.image_resolution

        images = [
            torch.zeros(batch_size, 3, height, width, dtype=torch.float32)
            for _ in range(num_cameras)
        ]
        img_masks = [
            torch.ones(batch_size, dtype=torch.bool) for _ in range(num_cameras)
        ]

        lang_tokens = torch.zeros(
            batch_size, config.tokenizer_max_length, dtype=torch.long
        )
        lang_masks = torch.ones(
            batch_size, config.tokenizer_max_length, dtype=torch.bool
        )

        state = torch.zeros(batch_size, config.max_state_dim, dtype=torch.float32)

        noise = self.policy.model.sample_noise(
            (batch_size, config.chunk_size, config.max_action_dim),
            device=state.device,
        )

        return images, img_masks, lang_tokens, lang_masks, state, noise
