# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


def patch_qwen3_vl_vision_forward(model):
    """Patch the Qwen3 VL vision encoder to compute pos embeds on CPU.

    In compile-only mode, XLA tensors have no real data. The vision encoder's
    fast_pos_embed_interpolate and rot_pos_emb call grid_thw.tolist() which
    returns zeros on XLA. We patch the vision forward to run entirely on CPU
    and transfer results back.
    """
    visual = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if visual is None:
        return

    original_forward = (
        visual.forward.__wrapped__ if hasattr(visual.forward, "__wrapped__") else None
    )

    class VisionForwardWrapper(nn.Module):
        def __init__(self, visual_module):
            super().__init__()
            self.visual = visual_module
            self._original_forward = visual_module.forward

        def __call__(self, hidden_states, grid_thw, **kwargs):
            input_device = hidden_states.device
            input_dtype = hidden_states.dtype
            hidden_states_cpu = hidden_states.detach().cpu().to(torch.float32)
            grid_thw_cpu = grid_thw.detach().cpu()

            visual_cpu = self.visual.cpu().float()
            with torch.no_grad():
                result = self.visual._original_cpu_forward(
                    hidden_states_cpu, grid_thw_cpu, **kwargs
                )

            if hasattr(result, "last_hidden_state"):
                out = result.last_hidden_state.to(
                    dtype=input_dtype, device=input_device
                )
                result.last_hidden_state = out
            elif isinstance(result, torch.Tensor):
                result = result.to(dtype=input_dtype, device=input_device)

            return result

    import types
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    if not hasattr(Qwen3VLVisionModel, "_original_cpu_forward"):
        Qwen3VLVisionModel._original_cpu_forward = Qwen3VLVisionModel.forward

    @torch.compiler.disable
    def cpu_vision_forward(self, hidden_states, grid_thw, **kwargs):
        input_device = hidden_states.device
        input_dtype = hidden_states.dtype

        hs_cpu = hidden_states.detach().cpu().to(torch.float32)
        grid_cpu = grid_thw.detach().cpu()

        self_cpu = self.cpu().float()
        result = self_cpu._original_cpu_forward(hs_cpu, grid_cpu, **kwargs)

        if hasattr(result, "last_hidden_state"):
            result.last_hidden_state = result.last_hidden_state.to(
                dtype=input_dtype, device=input_device
            )
        elif isinstance(result, torch.Tensor):
            result = result.to(dtype=input_dtype, device=input_device)
        elif isinstance(result, tuple):
            result = tuple(
                r.to(dtype=input_dtype, device=input_device)
                if isinstance(r, torch.Tensor)
                else r
                for r in result
            )

        return result

    visual.forward = types.MethodType(cpu_vision_forward, visual)
