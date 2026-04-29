# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Custom nn.Module for the FLUX.2 Fun ControlNet Union architecture.

The checkpoint format uses:
  control_img_in.{weight,bias}
  control_transformer_blocks.{i}.{before_proj, after_proj, attn, ff, ff_context}

Each block has joint attention (image + encoder streams) with RMS-normed QK,
and gated-GELU FFNs. No LayerNorm inside the blocks; normalization is expected
to be performed by the main transformer before passing hidden states in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FunControlNetFFN(nn.Module):
    """Gated-GELU FFN matching ff.linear_in / ff.linear_out checkpoint keys."""

    def __init__(self, dim: int, mult: int = 6):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim * mult, bias=False)
        self.linear_out = nn.Linear(dim * mult // 2, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.linear_in(x)
        gate, hidden = proj.chunk(2, dim=-1)
        return self.linear_out(F.gelu(hidden, approximate="tanh") * gate)


class _FunControlNetAttn(nn.Module):
    """Joint attention for image and encoder streams, matching checkpoint keys."""

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        # Stored as a ModuleList to match "attn.to_out.0.weight"
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.add_q_proj = nn.Linear(dim, dim, bias=False)
        self.add_k_proj = nn.Linear(dim, dim, bias=False)
        self.add_v_proj = nn.Linear(dim, dim, bias=False)
        self.to_add_out = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        B, img_seq, _ = hidden_states.shape
        txt_seq = encoder_hidden_states.shape[1]

        q = self.to_q(hidden_states).view(B, img_seq, self.num_heads, self.head_dim)
        k = self.to_k(hidden_states).view(B, img_seq, self.num_heads, self.head_dim)
        v = self.to_v(hidden_states).view(B, img_seq, self.num_heads, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)

        enc_q = self.add_q_proj(encoder_hidden_states).view(B, txt_seq, self.num_heads, self.head_dim)
        enc_k = self.add_k_proj(encoder_hidden_states).view(B, txt_seq, self.num_heads, self.head_dim)
        enc_v = self.add_v_proj(encoder_hidden_states).view(B, txt_seq, self.num_heads, self.head_dim)
        enc_q = self.norm_added_q(enc_q)
        enc_k = self.norm_added_k(enc_k)

        # Concatenate image and text for joint attention
        full_q = torch.cat([q, enc_q], dim=1).transpose(1, 2)  # [B, heads, seq, head_dim]
        full_k = torch.cat([k, enc_k], dim=1).transpose(1, 2)
        full_v = torch.cat([v, enc_v], dim=1).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(full_q, full_k, full_v)
        attn_out = attn_out.transpose(1, 2)  # [B, seq, heads, head_dim]

        img_attn = attn_out[:, :img_seq].reshape(B, img_seq, self.num_heads * self.head_dim)
        txt_attn = attn_out[:, img_seq:].reshape(B, txt_seq, self.num_heads * self.head_dim)

        return self.to_out[0](img_attn), self.to_add_out(txt_attn)


class _FunControlNetBlock(nn.Module):
    """Single ControlNet block with optional before_proj, after_proj, and joint attention.

    Only the first block (index 0) has a before_proj that injects the main transformer's
    hidden states into the control stream. All blocks have an after_proj for the residual.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, has_before_proj: bool = False):
        super().__init__()
        if has_before_proj:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)
        self.attn = _FunControlNetAttn(dim, num_heads, head_dim)
        self.ff = _FunControlNetFFN(dim)
        self.ff_context = _FunControlNetFFN(dim)

    def forward(
        self,
        ctrl_hidden: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        main_hidden: torch.Tensor,
    ):
        if hasattr(self, "before_proj"):
            h = ctrl_hidden + self.before_proj(main_hidden)
        else:
            h = ctrl_hidden
        img_attn, txt_attn = self.attn(h, encoder_hidden_states)
        h = h + img_attn
        enc = encoder_hidden_states + txt_attn
        h = h + self.ff(h)
        enc = enc + self.ff_context(enc)
        return self.after_proj(h), enc


class FluxFunControlNetModel(nn.Module):
    """
    FLUX.2 Fun ControlNet Union model.

    Loads directly from the alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union safetensors
    checkpoint. Produces per-block residuals intended to be injected into the first
    N double-stream blocks of a FLUX.2 transformer.

    Args:
        inner_dim:         Model hidden dimension (6144 for FLUX.2-dev).
        num_heads:         Number of attention heads (48 for FLUX.2-dev).
        head_dim:          Per-head dimension (128 for FLUX.2-dev).
        num_control_layers: Number of ControlNet blocks (4).
        in_channels:       Control-conditioning input dimension (260).
    """

    def __init__(
        self,
        inner_dim: int = 6144,
        num_heads: int = 48,
        head_dim: int = 128,
        num_control_layers: int = 4,
        in_channels: int = 260,
    ):
        super().__init__()
        self.control_img_in = nn.Linear(in_channels, inner_dim)
        self.control_transformer_blocks = nn.ModuleList([
            _FunControlNetBlock(inner_dim, num_heads, head_dim, has_before_proj=(i == 0))
            for i in range(num_control_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:         Image latent tokens [B, img_seq, inner_dim].
            encoder_hidden_states: Text tokens [B, txt_seq, inner_dim].
            controlnet_cond:       Control conditioning [B, img_seq, in_channels].

        Returns:
            Stacked block residuals [num_blocks, B, img_seq, inner_dim].
        """
        ctrl = self.control_img_in(controlnet_cond)
        residuals = []
        for block in self.control_transformer_blocks:
            residual, encoder_hidden_states = block(ctrl, encoder_hidden_states, hidden_states)
            ctrl = ctrl + residual
            residuals.append(residual)
        return torch.stack(residuals, dim=0)
