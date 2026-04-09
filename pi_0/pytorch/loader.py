# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi-0 model loader implementation for action prediction tasks
"""
import sys
import torch
import psutil
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


def _log_memory(tag: str) -> None:
    """Log host RAM usage at a checkpoint so OOM kills leave a breadcrumb trail.

    Prints immediately with flush=True so the line survives if the process is
    subsequently killed by the OOM killer (exit-code 137).
    """
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1024**3
    print(
        f"[PI0-MEM] {tag}: "
        f"system {vm.total / 1024**3:.1f} GB total, "
        f"{vm.available / 1024**3:.1f} GB available ({vm.percent}% used), "
        f"process RSS {rss_gb:.2f} GB",
        flush=True,
    )


class ModelVariant(StrEnum):
    """Available Pi-0 model variants."""

    LIBERO_BASE = "lerobot_pi0_libero_base"
    BASE = "pi0_base"


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
        return ModelInfo(
            model="pi_0",
            variant=variant,
            group=ModelGroup.RED,
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

        _log_memory("before model load")
        self.pretrained_model_name = self._variant_config.pretrained_model_name
        self.pi_0 = get_custom_pi0_policy(self.pretrained_model_name)
        self.pi_0.eval()
        _log_memory("after model load + eval")
        return self.pi_0

    def load_inputs(self, dtype_override=None, episode_index=0):
        """
        Load and preprocess inputs for action sampling.
        Returns images, image masks, language tokens, language masks, state,
        and a pre-generated noise tensor for deterministic diffusion sampling.
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from .src.model import preprocess_for_sampling

        _log_memory("before load_inputs")

        self.preprocess, self.postprocess_fn = make_pre_post_processors(
            self.pi_0.config,
            self.pretrained_model_name,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        _log_memory("after make_pre_post_processors")

        dataset = LeRobotDataset("lerobot/libero")
        _log_memory("after LeRobotDataset load")

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

        _log_memory("after load_inputs complete")
        return images, img_masks, lang_tokens, lang_masks, state, noise

    def postprocess(self, pred_action):
        """Apply postprocessing to predicted actions."""
        return self.postprocess_fn(pred_action)
