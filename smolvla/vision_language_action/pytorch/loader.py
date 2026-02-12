# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolVLA model loader implementation for action prediction.
"""

import importlib.util
import sys
import types
from typing import Optional

import numpy as np
import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _setup_policies_namespace() -> None:
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        return
    policies_path = Path(spec.origin).resolve().parent / "policies"
    if not policies_path.exists():
        return
    if "lerobot.policies" in sys.modules:
        return
    policies_module = types.ModuleType("lerobot.policies")
    policies_module.__path__ = [str(policies_path)]
    sys.modules["lerobot.policies"] = policies_module


from pathlib import Path

_setup_policies_namespace()

from lerobot.configs.types import FeatureType
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.utils import prepare_observation_for_inference
import lerobot.policies.smolvla.processor_smolvla  # Registers SmolVLA processor steps.
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION


class ModelVariant(StrEnum):
    """Available SmolVLA model variants."""

    SMOLVLA_BASE = "smolvla_base"


class ModelLoader(ForgeModel):
    """SmolVLA model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.SMOLVLA_BASE: ModelConfig(
            pretrained_model_name="lerobot/smolvla_base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOLVLA_BASE

    sample_task = "pick the red block"
    robot_type = ""

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.postprocess = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SmolVLA",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processors(self, device: torch.device):
        self.preprocess = PolicyProcessorPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": str(device)}},
        )
        self.postprocess = PolicyProcessorPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            config_filename="policy_postprocessor.json",
            overrides={"device_processor": {"device": str(device)}},
        )

    def load_model(self, *, dtype_override=None, device: str = "cpu", **kwargs):
        model = SmolVLAPolicy.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.to(device)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()
        self.config = model.config
        if self.preprocess is None or self.postprocess is None:
            self._load_processors(torch.device(device))
        return model

    def load_inputs(
        self, dtype_override=None, batch_size: int = 1, device: str = "cpu"
    ):
        if self.config is None:
            self.config = SmolVLAConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        if self.preprocess is None or self.postprocess is None:
            self._load_processors(torch.device(device))

        dummy_observation = build_dummy_observation(self.config.input_features or {})
        obs_frame = prepare_observation_for_inference(
            observation=dummy_observation,
            device=torch.device(device),
            task=self.sample_task,
            robot_type=self.robot_type,
        )

        action_dim = (
            self.config.action_feature.shape[0]
            if self.config.action_feature is not None
            else self.config.max_action_dim
        )
        action_dtype = dtype_override or torch.float32
        obs_frame[ACTION] = torch.zeros(
            (1, self.config.chunk_size, action_dim),
            dtype=action_dtype,
            device=device,
        )

        inputs = self.preprocess(obs_frame)

        if batch_size > 1:
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    inputs[key] = value.repeat_interleave(batch_size, dim=0)

        # SmolVLAPolicy.forward() expects (batch, noise=None, time=None, reduction="mean").
        # The test runner calls model(**load_inputs()), so we return {"batch": batch_dict}
        # so that model(batch=inputs) matches forward(batch, ...).
        return {"batch": inputs}

    def unpack_forward_output(self, fwd_output):
        """Extract the loss tensor from SmolVLAPolicy.forward() output (loss, loss_dict)."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]  # loss tensor
        return fwd_output


def build_dummy_observation(input_features: dict) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for key, feature in input_features.items():
        if not key.startswith("observation."):
            continue
        if feature.type == FeatureType.VISUAL:
            channels, height, width = feature.shape
            observation[key] = np.zeros((height, width, channels), dtype=np.uint8)
        elif feature.type in (FeatureType.STATE, FeatureType.ENV):
            observation[key] = np.zeros(feature.shape, dtype=np.float32)
    return observation
