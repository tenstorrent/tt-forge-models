# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MambaVision model loader implementation for image classification.
"""
import sys
import types

# mamba_ssm requires CUDA to build; inject a pure-PyTorch stub when absent
try:
    import mamba_ssm  # noqa: F401
except ImportError:
    import torch
    import torch.nn.functional as F

    def _selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=None,
    ):
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        A = A.float()
        B = B.float()
        C = C.float()
        batch, dim, seq_len = u.shape
        dstate = A.shape[1]
        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        x = torch.zeros(batch, dim, dstate, dtype=torch.float32, device=u.device)
        ys = []
        for i in range(seq_len):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2)
        out = y if D is None else y + u * D.float().unsqueeze(-1)
        if z is not None:
            out = out * F.silu(z.float())
        out = out.to(dtype=dtype_in)
        if return_last_state:
            return out, x
        return out

    _mamba_ssm_pkg = types.ModuleType("mamba_ssm")
    _mamba_ssm_ops = types.ModuleType("mamba_ssm.ops")
    _mamba_ssm_iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    _mamba_ssm_iface.selective_scan_fn = _selective_scan_fn
    _mamba_ssm_iface.selective_scan_ref = _selective_scan_fn
    _mamba_ssm_ops.selective_scan_interface = _mamba_ssm_iface
    _mamba_ssm_pkg.ops = _mamba_ssm_ops
    sys.modules["mamba_ssm"] = _mamba_ssm_pkg
    sys.modules["mamba_ssm.ops"] = _mamba_ssm_ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = _mamba_ssm_iface

from transformers import AutoModelForImageClassification
from timm.data.transforms_factory import create_transform
from datasets import load_dataset
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MambaVision model variants for image classification."""

    MAMBAVISION_T_1K = "MambaVision-T-1K"


class ModelLoader(ForgeModel):
    """MambaVision model loader implementation for image classification tasks."""

    _VARIANTS = {
        ModelVariant.MAMBAVISION_T_1K: ModelConfig(
            pretrained_model_name="nvidia/MambaVision-T-1K",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAMBAVISION_T_1K

    # MambaVision supports any input resolution; use the HF model card default.
    input_resolution = (3, 224, 224)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MambaVision",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self._cached_model is None:
            self.load_model(dtype_override=dtype_override)

        model = self._cached_model

        transform = create_transform(
            input_size=self.input_resolution,
            is_training=False,
            mean=model.config.mean,
            std=model.config.std,
            crop_mode=model.config.crop_mode,
            crop_pct=model.config.crop_pct,
        )

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = transform(image).unsqueeze(0)
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
