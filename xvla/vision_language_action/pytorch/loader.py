# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
X-VLA model loader implementation for vision-language-action prediction.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
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
    """Register lerobot.policies in sys.modules so subpackage imports work when this loader
    is imported outside the normal lerobot package context (e.g. via tt-forge-models dynamic
    import). Without this, 'from lerobot.policies.xvla...' can fail with import errors.
    """
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


class XVLAInferenceWrapper(torch.nn.Module):
    """torch.compile-friendly wrapper around XVLAModel for inference.

    Inlines generate_actions and forward_vlm to eliminate graph breaks
    (self.eval(), .item()) and data-dependent boolean indexing that the
    XLA/TT backend cannot handle.  All image views are assumed valid.
    """

    def __init__(self, policy):
        super().__init__()
        self.xvla_model = policy.model
        self.num_denoising_steps = policy.config.num_denoising_steps

    def forward(
        self,
        input_ids: torch.Tensor,
        image_input: torch.Tensor,
        domain_id: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        m = self.xvla_model
        target_dtype = torch.bfloat16 if m.config.dtype == "bfloat16" else torch.float32
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        # Inline forward_vlm without mask checks / boolean indexing.
        batch_size, num_views = image_input.shape[:2]
        flat_images = image_input.flatten(0, 1)
        image_feats = m.vlm._encode_image(flat_images)
        tokens_per_view, hidden_dim = image_feats.shape[1:]
        image_features = image_feats.view(
            batch_size, num_views, tokens_per_view, hidden_dim
        )

        inputs_embeds = m.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = m.vlm._merge_input_ids_with_image_features(
            image_features[:, 0], inputs_embeds
        )
        enc_out = m.vlm.language_model.model.encoder(
            attention_mask=attention_mask, inputs_embeds=merged_embeds
        )[0]
        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        enc = {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

        # Inline denoising loop from generate_actions.
        action_dim = m.dim_action
        x1 = torch.randn(
            batch_size,
            m.chunk_size,
            action_dim,
            device=proprio.device,
            dtype=target_dtype,
        )
        action = torch.zeros_like(x1)
        steps = max(1, int(self.num_denoising_steps))
        for i in range(steps, 0, -1):
            t = torch.full(
                (batch_size,), i / steps, device=proprio.device, dtype=target_dtype
            )
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = m.action_space.preprocess(proprio, x_t)
            action = m.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return m.action_space.postprocess(action)


class ModelVariant(StrEnum):
    """Available X-VLA model variants."""

    XVLA_BASE = "xvla_base"


class ModelLoader(ForgeModel):
    """X-VLA model loader implementation for vision-language-action prediction tasks."""

    _VARIANTS = {
        ModelVariant.XVLA_BASE: ModelConfig(
            pretrained_model_name="lerobot/xvla-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XVLA_BASE

    sample_task = "pick the red block"
    robot_type = ""

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="X-VLA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processors(self, device: torch.device):
        _setup_policies_namespace()
        import lerobot.policies.xvla.processor_xvla  # noqa: F401
        from lerobot.processor import PolicyProcessorPipeline

        self.preprocess = PolicyProcessorPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": str(device)}},
        )

    def load_model(self, *, dtype_override=None, device: str = "cpu", **kwargs):
        _setup_policies_namespace()
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        self._policy = XVLAPolicy.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        self._policy.to(device)
        self._policy = self._policy.to(dtype=torch.float32)
        self._policy.eval()
        self.config = self._policy.config
        if self.preprocess is None:
            self._load_processors(torch.device(device))
        return XVLAInferenceWrapper(self._policy)

    def load_inputs(
        self, dtype_override=None, batch_size: int = 1, device: str = "cpu"
    ):
        _setup_policies_namespace()
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.utils import prepare_observation_for_inference

        if self.config is None:
            self.config = XVLAConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        if self.preprocess is None:
            self._load_processors(torch.device(device))

        dummy_observation = build_dummy_observation(self.config.input_features or {})
        obs_frame = prepare_observation_for_inference(
            observation=dummy_observation,
            device=torch.device(device),
            task=self.sample_task,
            robot_type=self.robot_type,
        )

        batch = self.preprocess(obs_frame)

        if batch_size > 1:
            for key, value in batch.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    batch[key] = value.repeat_interleave(batch_size, dim=0)

        model_inputs = self._policy._build_model_inputs(batch)

        return {
            "input_ids": model_inputs["input_ids"],
            "image_input": model_inputs["image_input"],
            "domain_id": model_inputs["domain_id"],
            "proprio": model_inputs["proprio"],
        }

    def unpack_forward_output(self, fwd_output):
        """predict_action_chunk returns action tensor (B, n_steps, action_dim) directly."""
        return fwd_output


def build_dummy_observation(input_features: dict) -> dict[str, np.ndarray]:
    from lerobot.configs.types import FeatureType

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
