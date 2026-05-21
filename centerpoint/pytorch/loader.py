# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterPoint model loader implementation for tt_forge_models.

The RPN + CenterHead variant operates on pre-computed BEV feature maps
(B, 64, 512, 512) and is the primary compilation target for TT hardware.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

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
from .src.model import (
    CenterPointRPNHead,
    get_single_input,
    load_model_with_weights,
)


@dataclass
class CPConfig(ModelConfig):
    source: ModelSource = ModelSource.CUSTOM


class ModelVariant(StrEnum):
    RPN_HEAD = "centerpoint_rpn_head"


class ModelLoader(ForgeModel):
    """CenterPoint (RPN + CenterHead) model loader.

    The model accepts BEV pseudo-images produced by a PillarFeatureNet +
    scatter step and outputs per-task heatmaps and regression heads.

    Input shape:  (B, 64, 512, 512)  — BEV feature map in bfloat16
    Output:       List[Dict[str, Tensor]]  — one dict per detection task
    """

    _VARIANTS: Dict[StrEnum, ModelConfig] = {
        ModelVariant.RPN_HEAD: CPConfig(pretrained_model_name="centerpoint"),
    }
    DEFAULT_VARIANT = ModelVariant.RPN_HEAD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CenterPoint",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(
        self, dtype_override: torch.dtype = torch.bfloat16
    ) -> CenterPointRPNHead:
        """Load CenterPointRPNHead with pretrained mmdetection3d weights.

        Args:
            dtype_override: Model parameter dtype (default bfloat16 for TT).

        Returns:
            Pretrained CenterPointRPNHead in eval mode.
        """
        model = load_model_with_weights(dtype=dtype_override)
        self._model = model
        return model

    def load_inputs(
        self,
        dtype_override: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
    ):
        """Return a synthetic BEV feature map as model input.

        Args:
            dtype_override: Tensor dtype (default bfloat16).
            batch_size:     Batch dimension.

        Returns:
            Tuple containing one BEV tensor (B, 64, 512, 512).
        """
        bev = get_single_input(dtype=dtype_override, batch_size=batch_size)
        return (bev,)

    def unpack_forward_output(self, fwd_output) -> torch.Tensor:
        """Flatten List[Dict[str, Tensor]] into a single tensor.

        The model returns one dict per detection task.  We concatenate all
        per-task tensors so that the training backward pass has a scalar
        loss target.

        Args:
            fwd_output: Output from CenterPointRPNHead.forward()

        Returns:
            Concatenated, flattened tensor of all head outputs.
        """
        if isinstance(fwd_output, list):
            tensors = []
            for task_dict in fwd_output:
                if isinstance(task_dict, dict):
                    for v in task_dict.values():
                        if isinstance(v, torch.Tensor):
                            tensors.append(v.flatten())
            if tensors:
                return torch.cat(tensors)
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()
        return fwd_output
