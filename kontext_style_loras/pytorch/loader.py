# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Kontext-dev style LoRA model loader implementation.

Instantiates FluxTransformer2DModel from the published FLUX.1-dev config with
random weights and applies style LoRA weights from Owen777/Kontext-Style-Loras.
The LoRA targets only attention layers (to_k, to_q, to_v) that are
architecturally identical between FLUX.1-dev and FLUX.1-Kontext-dev, so the
ungated FLUX.1-dev config can be used as the base architecture.

Available variants:
- KONTEXT_STYLE_3D_CHIBI: 3D Chibi style LoRA
- KONTEXT_STYLE_GHIBLI: Ghibli style LoRA
- KONTEXT_STYLE_VAN_GOGH: Van Gogh style LoRA
"""

from typing import Any, Optional

import torch
from diffusers import FluxTransformer2DModel

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

LORA_REPO = "Owen777/Kontext-Style-Loras"

# FLUX.1-dev transformer config (same attention architecture as FLUX.1-Kontext-dev).
# Source: camenduru/FLUX.1-dev-ungated transformer/config.json
_TRANSFORMER_CONFIG = {
    "attention_head_dim": 128,
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available Kontext-Style-Loras variants."""

    KONTEXT_STYLE_3D_CHIBI = "kontext_style_3d_chibi"
    KONTEXT_STYLE_GHIBLI = "kontext_style_ghibli"
    KONTEXT_STYLE_VAN_GOGH = "kontext_style_van_gogh"


_LORA_FILES = {
    ModelVariant.KONTEXT_STYLE_3D_CHIBI: "3D_Chibi_lora_weights.safetensors",
    ModelVariant.KONTEXT_STYLE_GHIBLI: "Ghibli_lora_weights.safetensors",
    ModelVariant.KONTEXT_STYLE_VAN_GOGH: "Van_Gogh_lora_weights.safetensors",
}


class ModelLoader(ForgeModel):
    """FLUX.1-Kontext-dev style LoRA model loader."""

    _VARIANTS = {
        ModelVariant.KONTEXT_STYLE_3D_CHIBI: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
        ModelVariant.KONTEXT_STYLE_GHIBLI: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
        ModelVariant.KONTEXT_STYLE_VAN_GOGH: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.KONTEXT_STYLE_3D_CHIBI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer: Optional[FluxTransformer2DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="KONTEXT_STYLE_LORAS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Build FluxTransformer2DModel from config with LoRA weights fused in.

        The base model is instantiated with random weights (no large checkpoint
        download) using the published FLUX.1-dev config. LoRA adapters target
        only attention layers shared between FLUX.1-dev and FLUX.1-Kontext-dev.

        Returns:
            FluxTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.transformer = FluxTransformer2DModel(**_TRANSFORMER_CONFIG).to(dtype=dtype)

        self.transformer.load_lora_adapter(
            LORA_REPO,
            weight_name=_LORA_FILES[self._variant],
            prefix="transformer",
        )
        self.transformer.fuse_lora()
        self.transformer.unload_lora()

        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

        return self.transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Prepare synthetic inputs for FluxTransformer2DModel.forward.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default 1).

        Returns:
            dict of input tensors matching FluxTransformer2DModel.forward.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.transformer.config

        height, width = 128, 128
        vae_scale_factor = 8
        max_seq_len = 256

        # Packed latent layout: each 2x2 spatial patch is flattened
        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2
        num_patches = h_packed * w_packed

        hidden_states = torch.randn(
            batch_size, num_patches, config.in_channels, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, max_seq_len, config.joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)

        img_ids = torch.zeros(h_packed, w_packed, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h_packed)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w_packed)[None, :]
        img_ids = img_ids.reshape(-1, 3)

        txt_ids = torch.zeros(max_seq_len, 3)

        guidance = (
            torch.full([batch_size], 3.5, dtype=dtype)
            if config.guidance_embeds
            else None
        )

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,
        }
