# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moirai model loader implementation for time series forecasting.
"""

from dataclasses import dataclass
from typing import Optional

import torch

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


@dataclass
class MoiraiConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 64
    patch_size: int = 32
    target_dim: int = 1
    num_samples: int = 100


class ModelVariant(StrEnum):
    """Available Moirai model variants."""

    BASE_1_0 = "base_1_0"
    LARGE = "large"
    LARGE_1_0 = "large_1_0"


class ModelLoader(ForgeModel):
    """Moirai model loader for time series forecasting.

    Uses the uni2ts MoiraiForecast wrapper around MoiraiModule
    for univariate time series prediction.
    """

    _VARIANTS = {
        ModelVariant.BASE_1_0: MoiraiConfig(
            pretrained_model_name="Salesforce/moirai-1.0-R-base",
            context_length=512,
            prediction_length=64,
            patch_size=32,
            target_dim=1,
            num_samples=100,
        ),
        ModelVariant.LARGE: MoiraiConfig(
            pretrained_model_name="Salesforce/moirai-1.1-R-large",
            context_length=512,
            prediction_length=64,
            patch_size=32,
            target_dim=1,
            num_samples=100,
        ),
        ModelVariant.LARGE_1_0: MoiraiConfig(
            pretrained_model_name="Salesforce/moirai-1.0-R-large",
            context_length=512,
            prediction_length=64,
            patch_size=32,
            target_dim=1,
            num_samples=100,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Moirai",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        from einops import reduce, repeat
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        cfg = self._variant_config

        module = MoiraiModule.from_pretrained(cfg.pretrained_model_name)

        model = MoiraiForecast(
            module=module,
            prediction_length=cfg.prediction_length,
            context_length=cfg.context_length,
            patch_size=cfg.patch_size,
            target_dim=cfg.target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            num_samples=cfg.num_samples,
        )

        # Replace cummax (→ stablehlo.reduce_window, unsupported in TT MLIR) with
        # the equivalent for binary {0,1} sequences: cummax(x) == (cumsum(x) > 0)
        def _generate_time_id_patched(self_m, patch_size, past_observed_target):
            past_seq_id = reduce(
                self_m._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
                "... (seq patch) dim -> ... seq",
                "max",
                patch=patch_size,
            )
            past_seq_id = torch.clamp(
                (past_seq_id.cumsum(dim=-1) > 0).to(past_seq_id.dtype).cumsum(dim=-1) - 1,
                min=0,
            )
            batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
            future_seq_id = (
                repeat(
                    torch.arange(
                        self_m.prediction_token_length(patch_size),
                        device=past_observed_target.device,
                    ),
                    f"prediction -> {batch_shape} prediction",
                )
                + past_seq_id.max(dim=-1, keepdim=True).values
                + 1
            )
            return past_seq_id, future_seq_id

        import types
        model._generate_time_id = types.MethodType(_generate_time_id_patched, model)

        model.eval()
        return model

    def load_inputs(self):
        cfg = self._variant_config

        torch.manual_seed(42)

        past_target = torch.randn(1, cfg.context_length, cfg.target_dim)
        # Use int32 instead of bool: XLA cumsum(bool) returns bool (not int64 like CPU),
        # causing the cummax→cumsum→sub chain in _generate_time_id to fail at compile time.
        past_observed_target = torch.ones(
            1, cfg.context_length, cfg.target_dim, dtype=torch.int32
        )
        past_is_pad = torch.zeros(1, cfg.context_length, dtype=torch.int32)

        return {
            "past_target": past_target,
            "past_observed_target": past_observed_target,
            "past_is_pad": past_is_pad,
        }
