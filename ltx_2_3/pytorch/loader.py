# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 model loader.

LTX-2.3 is a 22B DiT (Diffusion Transformer) video generation model by Lightricks.
Repository: https://github.com/Lightricks/LTX-2
Weights:    https://huggingface.co/Lightricks/LTX-2.3

Variants
--------
LTX_23_FAST  —  distilled checkpoint, 8 steps, guidance_scale=1.0 (CFG off)
LTX_23_PRO   —  dev checkpoint, ~40 steps, guidance_scale=5.0 (CFG on)

Both variants share the same 22B DiT architecture. Only the checkpoint weights,
inference step count, and guidance settings differ.

Supported subfolders
--------------------
None          — returns the raw transformer nn.Module loaded from the checkpoint
"transformer" — same as None (kept for API symmetry with other loaders)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
from .src.utils import (
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WIDTH,
    load_transformer_direct,
    load_transformer_inputs,
)

_SUPPORTED_SUBFOLDERS = {None, "transformer"}
_HF_REPO = "Lightricks/LTX-2.3"


@dataclass
class LTX23Config(ModelConfig):
    checkpoint: str = ""
    pipeline_class: str = "DistilledPipeline"
    num_inference_steps: int = 8
    guidance_scale: float = 1.0


class ModelVariant(StrEnum):
    LTX_23_FAST = "ltx-2.3-22b-distilled-1.1"
    LTX_23_PRO = "ltx-2.3-22b-dev"


class ModelLoader(ForgeModel):
    """
    Loader for LTX-2.3 Fast and Pro variants.

    Typical usage (transformer compilation target)::

        loader = ModelLoader(ModelVariant.LTX_23_FAST, subfolder="transformer")
        model  = loader.load_model()   # returns the DiT nn.Module
        inputs = loader.load_inputs()  # returns dict of synthetic tensors
    """

    _VARIANTS = {
        ModelVariant.LTX_23_FAST: LTX23Config(
            pretrained_model_name=_HF_REPO,
            checkpoint="ltx-2.3-22b-distilled-1.1.safetensors",
            pipeline_class="DistilledPipeline",
            num_inference_steps=8,
            guidance_scale=1.0,
        ),
        ModelVariant.LTX_23_PRO: LTX23Config(
            pretrained_model_name=_HF_REPO,
            checkpoint="ltx-2.3-22b-dev.safetensors",
            pipeline_class="TI2VidOneStagePipeline",
            num_inference_steps=40,
            guidance_scale=5.0,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LTX_23_FAST

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = None,
    ):
        super().__init__(variant)
        if subfolder not in _SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder '{subfolder}'. Supported: {_SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX-2.3",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs) -> torch.nn.Module:
        cfg: LTX23Config = self._variant_config
        transformer = load_transformer_direct(cfg.pretrained_model_name, cfg.checkpoint)
        transformer.eval()
        return transformer

    def load_inputs(
        self,
        dtype: torch.dtype = torch.bfloat16,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        cfg: LTX23Config = self._variant_config
        return load_transformer_inputs(
            dtype=dtype,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=cfg.guidance_scale,
        )

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
