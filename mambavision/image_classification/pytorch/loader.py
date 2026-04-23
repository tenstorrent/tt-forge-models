# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MambaVision model loader implementation for image classification.
"""
import sys
import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification
from timm.data.transforms_factory import create_transform
from datasets import load_dataset
from typing import Optional


def _install_mamba_ssm_stub():
    """Inject a pure-PyTorch stub for mamba_ssm (real package requires CUDA)."""
    if "mamba_ssm" in sys.modules:
        return

    def selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, seqlen = u.shape
        B_f = B.float()
        C_f = C.float()
        x = A.new_zeros((batch, dim, A.shape[1]))
        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A.float()))
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B_f, u)
        ys = []
        last_state = None
        for i in range(seqlen):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = torch.einsum("bdn,bn->bd", x, C_f[:, :, i])
            if i == seqlen - 1:
                last_state = x
            ys.append(y)
        y = torch.stack(ys, dim=2)
        if D is not None:
            y = y + u * D.float().unsqueeze(-1)
        if z is not None:
            y = y * F.silu(z.float())
        out = y.to(dtype=dtype_in)
        if return_last_state:
            return out, last_state
        return out

    interface_mod = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    interface_mod.selective_scan_fn = selective_scan_fn
    interface_mod.selective_scan_ref = selective_scan_fn

    ops_mod = types.ModuleType("mamba_ssm.ops")
    ops_mod.selective_scan_interface = interface_mod

    root_mod = types.ModuleType("mamba_ssm")
    root_mod.ops = ops_mod

    sys.modules["mamba_ssm"] = root_mod
    sys.modules["mamba_ssm.ops"] = ops_mod
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = interface_mod


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
        _install_mamba_ssm_stub()
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
