# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the OLM-OCR (Qwen2.5-VL) loader."""


def _install_dynamo_safe_param_props(model):
    """Make dtype/device iterate named_parameters() instead of parameters(), on
    the model and every submodule that exposes those properties.

    torch 2.10 Dynamo's wrap_values() emits an invalid `named_children` reference
    when tracing .parameters(); named_parameters() takes a clean path. The
    Qwen2.5-VL forward reads `self.visual.dtype` (a PreTrainedModel property that
    calls `next(p.dtype for p in self.parameters() ...)`) to cast pixel_values,
    so the vision tower and every PreTrainedModel in the tree must be patched,
    not just the root.
    """
    for module in list(model.modules()):
        cls = type(module)
        if getattr(cls, "_tt_xla_dynamo_safe", False):
            continue
        # Only patch modules that actually expose the buggy property.
        if not (
            isinstance(getattr(cls, "dtype", None), property)
            or isinstance(getattr(cls, "device", None), property)
        ):
            continue

        class _DynamoSafe(cls):
            _tt_xla_dynamo_safe = True

            @property
            def dtype(self):
                for _, p in self.named_parameters():
                    if p.is_floating_point():
                        return p.dtype
                return super().dtype

            @property
            def device(self):
                for _, p in self.named_parameters():
                    return p.device
                return super().device

        module.__class__ = _DynamoSafe
    return model
