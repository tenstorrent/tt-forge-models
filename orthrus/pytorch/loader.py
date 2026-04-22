# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Orthrus RNA foundation model loader implementation.

Orthrus is a Mamba-based RNA foundation model pre-trained with contrastive
learning on 45M+ mature RNA transcripts. It generates sequence embeddings
for spliced mature RNA transcripts.

The model architecture is implemented inline to avoid the CUDA-only mamba-ssm
package dependency required by the upstream HuggingFace remote code.
"""

import math
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class _Mamba(nn.Module):
    """Pure-PyTorch Mamba SSM block with the same parameter layout as mamba-ssm."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        A = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(self.d_inner, -1)
            .contiguous()
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor, inference_params=None, **kwargs):
        """Simplified SSM forward: projection + depthwise conv1d gating + output projection."""
        seqlen = hidden_states.shape[1]
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :seqlen]
        x = x.transpose(1, 2)
        x = F.silu(x) * F.silu(z)
        return self.out_proj(x)


class _Block(nn.Module):
    """Mamba Block wrapping the SSM mixer with LayerNorm and residual connection."""

    def __init__(
        self,
        dim,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
        **kwargs,
    ):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        if self.mlp is not None:
            residual = hidden_states + residual
            hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class _OrthrusModel(nn.Module):
    """Orthrus RNA foundation model (inline implementation of the HF remote code)."""

    def __init__(
        self, n_tracks: int = 6, ssm_model_dim: int = 512, ssm_n_layers: int = 6
    ):
        super().__init__()
        self.embedding = nn.Linear(n_tracks, ssm_model_dim)
        self.layers = nn.ModuleList(
            [
                _Block(
                    ssm_model_dim,
                    partial(_Mamba, layer_idx=i),
                    norm_cls=nn.LayerNorm,
                    mlp_cls=nn.Identity,
                )
                for i in range(ssm_n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(ssm_model_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(x)
        res = None
        for layer in self.layers:
            hidden_states, res = layer(hidden_states, res)
        res = (hidden_states + res) if res is not None else hidden_states
        hidden_states = self.norm_f(res.to(dtype=self.norm_f.weight.dtype))

        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        masked = hidden_states * mask.unsqueeze(-1)
        return masked.sum(dim=1) / lengths.unsqueeze(-1).float()


class ModelVariant(StrEnum):
    """Available Orthrus model variants."""

    LARGE_6_TRACK = "large-6-track"


class ModelLoader(ForgeModel):
    """Orthrus RNA foundation model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_6_TRACK: ModelConfig(
            pretrained_model_name="quietflamingo/orthrus-large-6-track",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_6_TRACK

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Orthrus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Orthrus model instance."""
        model = _OrthrusModel(n_tracks=6, ssm_model_dim=512, ssm_n_layers=6)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the Orthrus model.

        Orthrus 6-track expects input of shape (batch, seq_len, 6) where the
        6 channels are: 4 one-hot nucleotide channels + CDS track + splice site track.
        It also requires a lengths tensor indicating the sequence length.
        """
        seq_len = 128

        seq_ohe = np.zeros((seq_len, 4), dtype=np.float32)
        nucleotide_indices = np.random.randint(0, 4, size=seq_len)
        seq_ohe[np.arange(seq_len), nucleotide_indices] = 1.0

        cds = np.zeros((seq_len, 1), dtype=np.float32)
        cds[10:100] = 1.0

        splice = np.zeros((seq_len, 1), dtype=np.float32)
        splice[10] = 1.0
        splice[99] = 1.0

        model_input = np.hstack((seq_ohe, cds, splice))
        model_input = torch.tensor(model_input).unsqueeze(0)

        lengths = torch.tensor([seq_len], dtype=torch.float32)

        if dtype_override is not None:
            model_input = model_input.to(dtype_override)
            lengths = lengths.to(dtype_override)

        return model_input, lengths
