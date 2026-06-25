# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi-0.5 (Pi05) model loader implementation for action prediction tasks.
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
    """Pi-0.5 model loader implementation for the action prediction task."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: Pi05ModelConfig(
            pretrained_model_name="lerobot/pi05_base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Vision configuration (shared across all variants): 3 camera views at
    # 224x224, square SigLIP resolution.
    NUM_CAMERAS = 3
    IMAGE_SIZE = 224
    # Language prompt length used for the synthetic inputs. The model's
    # configured maximum is 200; a realistic instruction is far shorter, so a
    # modest length keeps the prefix sequence (image patches + language) lean
    # without changing the architecture under test.
    LANGUAGE_LEN = 48
    # Conservative upper bound below the PaliGemma vocab (257152) for sampling
    # synthetic token ids.
    VOCAB_LIMIT = 256000

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pi_05 = None

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
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Pi-0.5 model instance.

        Args:
            dtype_override: Optional torch dtype to cast the model weights to
                (e.g. torch.bfloat16 for the device path). The checkpoint is
                distributed in float32.

        Returns:
            torch.nn.Module: The Pi-0.5 Policy instance with an inference
                ``forward`` that drives the flow-matching sampling loop.
        """
        from .src.model import get_custom_pi05_policy

        # Map a torch dtype override to the policy config's dtype string so the
        # model is *built* at that precision. Building directly in bf16 avoids
        # ever materializing the ~14.5 GB fp32 instance, which is what lets the
        # 4B-param model load + compile within a constrained host (e.g. n150's
        # 31 GB host OOMs on the fp32 path).
        _DTYPE_TO_STR = {torch.bfloat16: "bfloat16", torch.float32: "float32"}
        config_dtype = _DTYPE_TO_STR.get(dtype_override)

        self.pretrained_model_name = self._variant_config.pretrained_model_name
        self.pi_05 = get_custom_pi05_policy(
            self.pretrained_model_name, config_dtype=config_dtype
        )
        self.pi_05.eval()

        # Fallback cast for any dtype not expressible via the config string
        # (or for residual fp32 leaf modules such as the small action MLPs).
        if dtype_override is not None:
            self.pi_05 = self.pi_05.to(dtype_override)

        return self.pi_05

    def load_inputs(self, dtype_override=None):
        """Build deterministic synthetic inputs for action sampling.

        Returns a tuple matching the custom inference ``forward`` signature:
        (images, img_masks, tokens, masks, noise). Pi-0.5 encodes the robot
        state inside the language tokens, so there is no separate state input.

        Args:
            dtype_override: Optional torch dtype for the floating-point inputs
                (images and noise); must match the model dtype.

        Returns:
            tuple: (images, img_masks, tokens, masks, noise)
        """
        cfg = self.pi_05.config
        gen = torch.Generator().manual_seed(0)
        bsize = 1
        float_dtype = dtype_override or torch.float32

        # One RGB image per camera view, plus an all-valid view mask.
        images = [
            torch.rand(
                bsize, 3, self.IMAGE_SIZE, self.IMAGE_SIZE, generator=gen
            ).to(float_dtype)
            for _ in range(self.NUM_CAMERAS)
        ]
        img_masks = [
            torch.ones(bsize, dtype=torch.bool) for _ in range(self.NUM_CAMERAS)
        ]

        # State-encoded language tokens with an all-attended mask.
        tokens = torch.randint(
            0, self.VOCAB_LIMIT, (bsize, self.LANGUAGE_LEN), generator=gen
        )
        masks = torch.ones(bsize, self.LANGUAGE_LEN, dtype=torch.bool)

        # Deterministic starting noise for flow matching so CPU and device runs
        # are comparable.
        noise = torch.randn(
            bsize, cfg.chunk_size, cfg.max_action_dim, generator=gen
        ).to(float_dtype)

        return images, img_masks, tokens, masks, noise
