# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VQ-BeT PushT model loader for tt_forge_models.

VQ-BeT (Vector Quantized Behavior Transformer) combines a ResNet18 vision
backbone with a GPT-style transformer to predict quantized action tokens
for robotic control on the PushT environment. Actions are produced through
a VQ-VAE codebook.

Reference: https://huggingface.co/lerobot/vqbet_pusht
"""

from typing import Optional

import torch
from lerobot.policies.vqbet import VQBeTPolicy

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
    """Available VQ-BeT PushT model variants."""

    PUSHT = "pusht"


class ModelLoader(ForgeModel):
    """VQ-BeT PushT model loader.

    Loads the VQ-BeT policy trained on the PushT environment for robotic
    action prediction. The model consumes RGB images and 2D agent state
    across a short observation history and emits 2D agent actions.
    """

    _VARIANTS = {
        ModelVariant.PUSHT: ModelConfig(
            pretrained_model_name="lerobot/vqbet_pusht",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PUSHT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.policy = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VQBeTPushT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.policy = VQBeTPolicy.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        self.policy.eval()

        return self.policy

    def load_inputs(self, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32

        batch_size = 1
        n_obs_steps = 5

        # observation.image: 3x96x96 RGB frames of the PushT environment
        image = torch.rand((batch_size, n_obs_steps, 3, 96, 96), dtype=dtype)
        # observation.state: 2-dim agent position (x, y)
        state = torch.randn((batch_size, n_obs_steps, 2), dtype=dtype)

        return {
            "observation.image": image,
            "observation.state": state,
        }

    def unpack_forward_output(self, output):
        if isinstance(output, dict) and "action" in output:
            return output["action"]
        elif isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, tuple):
            return output[0]
        return output
