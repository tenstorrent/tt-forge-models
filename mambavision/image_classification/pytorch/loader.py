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
    """Pure-PyTorch selective scan for CPU/XLA tracing (mamba_ssm CUDA fallback)."""
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


def _ensure_mamba_ssm_stub():
    """Inject a pure-PyTorch mamba_ssm stub so modeling_mambavision.py can load.

    mamba_ssm requires CUDA extensions that cannot be installed in CPU/XLA
    environments.  We register fake module objects in sys.modules so that
    ``from mamba_ssm.ops.selective_scan_interface import selective_scan_fn``
    resolves correctly regardless of whether the real package is present.
    """
    iface_key = "mamba_ssm.ops.selective_scan_interface"
    iface = sys.modules.get(iface_key)
    if iface is not None and hasattr(iface, "selective_scan_fn"):
        return

    _pkg = types.ModuleType("mamba_ssm")
    _ops = types.ModuleType("mamba_ssm.ops")
    _ops.__path__ = []
    _iface = types.ModuleType(iface_key)
    _iface.selective_scan_fn = _selective_scan_fn
    _iface.selective_scan_ref = _selective_scan_fn
    _ops.selective_scan_interface = _iface
    _pkg.ops = _ops
    _pkg.__path__ = []
    sys.modules.setdefault("mamba_ssm", _pkg)
    sys.modules.setdefault("mamba_ssm.ops", _ops)
    sys.modules[iface_key] = _iface


_ensure_mamba_ssm_stub()

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
        _ensure_mamba_ssm_stub()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # MambaVision calls .item() on a torch.linspace() result during __init__,
        # which fails when the model is initialized inside a torch.device("meta")
        # context (used by transformers 5.x always).  Patch get_init_context to
        # remove the meta-device context so .item() works on real CPU tensors.
        from transformers import PreTrainedModel

        _orig_get_init_context = PreTrainedModel.get_init_context.__func__

        @classmethod
        def _no_meta_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            contexts = _orig_get_init_context(
                cls, dtype, is_quantized, _is_ds_init_called
            )
            return [
                c
                for c in contexts
                if not (isinstance(c, torch.device) and c.type == "meta")
            ]

        PreTrainedModel.get_init_context = _no_meta_init_context
        try:
            model = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            PreTrainedModel.get_init_context = classmethod(_orig_get_init_context)
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
