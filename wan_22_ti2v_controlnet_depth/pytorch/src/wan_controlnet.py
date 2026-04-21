# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Vendored WanControlnet model definition.

Sourced from https://github.com/TheDenk/wan2.2-controlnet (Apache-2.0) because
the ``WanControlnet`` class is not available in upstream ``diffusers``.
Only the class definition is vendored; loading is driven by diffusers'
``ModelMixin.from_pretrained`` via the accompanying ``model_utils.py``.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_wan import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.get_logger(__name__)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class WanControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """A ControlNet Transformer for the Wan video diffusion family."""

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 3,
        vae_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 20,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        downscale_coef: int = 8,
        out_proj_dim: int = 128 * 12,
    ) -> None:
        super().__init__()

        start_channels = in_channels * (downscale_coef**2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4]

        self.control_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        input_channels[0],
                        kernel_size=(3, downscale_coef + 1, downscale_coef + 1),
                        stride=(1, downscale_coef, downscale_coef),
                        padding=(1, downscale_coef // 2, downscale_coef // 2),
                    ),
                    nn.GELU(approximate="tanh"),
                    nn.GroupNorm(2, input_channels[0]),
                ),
                nn.Sequential(
                    nn.Conv3d(
                        input_channels[0],
                        input_channels[1],
                        kernel_size=3,
                        stride=(2, 1, 1),
                        padding=1,
                    ),
                    nn.GELU(approximate="tanh"),
                    nn.GroupNorm(2, input_channels[1]),
                ),
                nn.Sequential(
                    nn.Conv3d(
                        input_channels[1],
                        input_channels[2],
                        kernel_size=3,
                        stride=(2, 1, 1),
                        padding=1,
                    ),
                    nn.GELU(approximate="tanh"),
                    nn.GroupNorm(2, input_channels[2]),
                ),
            ]
        )

        inner_dim = num_attention_heads * attention_head_dim

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            vae_channels + input_channels[2],
            inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.controlnet_blocks = nn.ModuleList(
            [zero_module(nn.Linear(inner_dim, out_proj_dim)) for _ in range(num_layers)]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif (
            attention_kwargs is not None
            and attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

        rotary_emb = self.rope(hidden_states)

        for control_encoder_block in self.control_encoder:
            controlnet_states = control_encoder_block(controlnet_states)
        hidden_states = torch.cat([hidden_states, controlnet_states], dim=1)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
        ) = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=ts_seq_len,
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        controlnet_hidden_states: Tuple[torch.Tensor, ...] = ()
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
                controlnet_hidden_states += (controlnet_block(hidden_states),)
        else:
            for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                controlnet_hidden_states += (controlnet_block(hidden_states),)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_hidden_states,)

        return Transformer2DModelOutput(sample=controlnet_hidden_states)
