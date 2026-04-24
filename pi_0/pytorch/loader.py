# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi-0 model loader implementation for action prediction tasks
"""
import torch
from typing import Optional, Dict, Any
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
    """Available Pi-0 model variants."""

    LIBERO_BASE = "lerobot_pi0_libero_base"
    BASE = "pi0_base"
    INTACT_FINETUNE_BRIDGE = "juexzz_intact_pi0_finetune_bridge"


class ModelLoader(ForgeModel):
    """Pi-0 model loader implementation for action prediction task."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LIBERO_BASE: ModelConfig(
            pretrained_model_name="lerobot/pi0_libero_base",
        ),
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="lerobot/pi0_base",
        ),
        ModelVariant.INTACT_FINETUNE_BRIDGE: ModelConfig(
            pretrained_model_name="juexzz/INTACT-pi0-finetune-bridge",
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
            variant_name: Optional variant name string.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant == ModelVariant.INTACT_FINETUNE_BRIDGE:
            group = ModelGroup.VULCAN
        else:
            group = ModelGroup.RED

        return ModelInfo(
            model="pi_0",
            variant=variant,
            group=group,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Pi-0 model instance with default settings.

        Returns:
            torch.nn.Module: The Pi-0 Policy instance.
        """

        from .src.model import get_custom_pi0_policy

        self.pretrained_model_name = self._variant_config.pretrained_model_name
        self.pi_0 = get_custom_pi0_policy(self.pretrained_model_name)
        self.pi_0.eval()
        return self.pi_0

    def _load_inputs_synthetic(self):
        """Create synthetic inputs directly from model config.

        Used when the preprocessing pipeline cannot be built (e.g. the
        PaLIGemma tokenizer is behind a gated HuggingFace repo that the
        current token cannot access).
        """
        cfg = self.pi_0.config
        bsize = 1
        img_h, img_w = cfg.image_resolution
        images = [torch.zeros(bsize, 3, img_h, img_w) for _ in cfg.image_features]
        img_masks = [torch.ones(bsize, dtype=torch.bool) for _ in cfg.image_features]
        lang_tokens = torch.zeros(bsize, cfg.tokenizer_max_length, dtype=torch.long)
        lang_masks = torch.ones(bsize, cfg.tokenizer_max_length, dtype=torch.bool)
        state = torch.zeros(bsize, cfg.max_state_dim)
        noise = self.pi_0.model.sample_noise(
            (bsize, cfg.chunk_size, cfg.max_action_dim),
            device=state.device,
        )
        return images, img_masks, lang_tokens, lang_masks, state, noise

    def load_inputs(self, dtype_override=None, episode_index=0):
        """
        Load and preprocess inputs for action sampling.
        Returns images, image masks, language tokens, language masks, state,
        and a pre-generated noise tensor for deterministic diffusion sampling.
        """
        from huggingface_hub import hf_hub_download
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from .src.model import preprocess_for_sampling

        try:
            hf_hub_download(
                repo_id=self.pretrained_model_name,
                filename="policy_preprocessor.json",
                local_files_only=True,
            )
            preprocessor_path = self.pretrained_model_name
        except Exception:
            preprocessor_path = None

        try:
            self.preprocess, self.postprocess_fn = make_pre_post_processors(
                self.pi_0.config,
                preprocessor_path,
                preprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
        except Exception:
            return self._load_inputs_synthetic()

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
        ) = self.pi_0.preprocess_for_sampling(batch)

        bsize = state.shape[0]
        noise = self.pi_0.model.sample_noise(
            (bsize, self.pi_0.config.chunk_size, self.pi_0.config.max_action_dim),
            device=state.device,
        )

        return images, img_masks, lang_tokens, lang_masks, state, noise

    def postprocess(self, pred_action):
        """Apply postprocessing to predicted actions."""
        return self.postprocess_fn(pred_action)
