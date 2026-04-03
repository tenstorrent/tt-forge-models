# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab RL policy loader — https://github.com/isaac-sim/IsaacLab

Locomotion policies trained with RSL-RL in NVIDIA Isaac Lab / Isaac Sim.
Each variant is an actor MLP that maps robot observations to joint position targets.

Observation spaces (concatenated vectors):
  AnymalC_Flat:  base vel(6) + gravity(3) + cmd(3) + joints(24) + prev_act(12) = 48
  AnymalC_Rough: flat(48) + height_scan(187) = 235
  H1_Velocity:   base vel(6) + gravity(3) + cmd(3) + joints(38) + prev_act(19) + height_scan(187) = 256
"""

import torch
from typing import Optional
from dataclasses import dataclass

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src.model import IsaacLabActorMLP

CHECKPOINT_URLS = {
    "anymalc_flat": (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
        "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Anymal-C-v0/checkpoint.pt"
    ),
    "anymalc_rough": (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
        "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-Anymal-C-v0/checkpoint.pt"
    ),
    "h1_velocity": (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
        "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-H1-v0/checkpoint.pt"
    ),
}


@dataclass
class IsaacLabPolicyConfig(ModelConfig):
    obs_dim: int = 48
    action_dim: int = 12
    hidden_dims: tuple = (128, 128, 128)


class ModelVariant(StrEnum):
    ANYMALC_FLAT = "AnymalC_Flat"
    ANYMALC_ROUGH = "AnymalC_Rough"
    H1_VELOCITY = "H1_Velocity"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.ANYMALC_FLAT: IsaacLabPolicyConfig(
            pretrained_model_name="anymalc_flat",
            obs_dim=48,
            action_dim=12,
            hidden_dims=(128, 128, 128),
        ),
        ModelVariant.ANYMALC_ROUGH: IsaacLabPolicyConfig(
            pretrained_model_name="anymalc_rough",
            obs_dim=235,
            action_dim=12,
            hidden_dims=(512, 256, 128),
        ),
        ModelVariant.H1_VELOCITY: IsaacLabPolicyConfig(
            pretrained_model_name="h1_velocity",
            obs_dim=256,
            action_dim=19,
            hidden_dims=(512, 256, 128),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANYMALC_FLAT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="IsaacLabPolicy",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        cfg = self._variant_config
        model = IsaacLabActorMLP(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            hidden_dims=cfg.hidden_dims,
        )
        self._try_load_pretrained(model, cfg.pretrained_model_name)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def _try_load_pretrained(self, model, variant_key):
        """Attempt to load NVIDIA Isaac Lab pretrained checkpoint.

        RSL-RL checkpoints store {'model_state_dict': ..., ...} where
        actor weights are keyed as 'actor.{idx}.weight/bias'.
        Falls back to random init if unavailable.
        """
        url = CHECKPOINT_URLS.get(variant_key)
        if url is None:
            return
        try:
            from ...tools.utils import get_file

            ckpt_path = get_file(url)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            actor_sd = {
                k: v for k, v in state_dict.items() if k.startswith("actor.")
            }
            if actor_sd:
                model.load_state_dict(actor_sd, strict=False)
        except Exception as e:
            print(f"[isaac_sim] Could not load pretrained weights for {variant_key}: {e}")
            print("[isaac_sim] Proceeding with random initialization.")

    def load_inputs(self, dtype_override=None, batch_size=1):
        cfg = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.float32
        obs = torch.randn(batch_size, cfg.obs_dim, dtype=dtype)
        return (obs,)
