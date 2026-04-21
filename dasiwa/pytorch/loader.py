# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

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

BASE_REPO_ID = "zai-org/GLM-Image"


class ModelVariant(StrEnum):
    DASIWA = "DASIWA"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.DASIWA: ModelConfig(
            pretrained_model_name="thatboymentor/DASIWA",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DASIWA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DASIWA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16

        if self._transformer is None:
            pipe = DiffusionPipeline.from_pretrained(
                BASE_REPO_ID,
                torch_dtype=dtype,
                **kwargs,
            )
            if hasattr(pipe, "load_lora_weights"):
                pipe.load_lora_weights(self._variant_config.pretrained_model_name)
                pipe.fuse_lora()
            self._transformer = pipe.transformer
            self._transformer.eval()
            del pipe

        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1) -> Any:
        dtype = dtype_override or torch.bfloat16

        # Config: in_channels=16, text_embed_dim=1472, patch_size=2
        # prior_vq_quantizer_codebook_size=16384
        in_channels = 16
        text_embed_dim = 1472
        patch_size = 2

        latent_h = 4
        latent_w = 4
        img_seq_len = (latent_h // patch_size) * (latent_w // patch_size)
        txt_seq_len = 8

        hidden_states = torch.randn(
            batch_size, img_seq_len, in_channels * patch_size * patch_size, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_embed_dim, dtype=dtype
        )
        prior_token_id = torch.randint(0, 16384, (batch_size,))
        prior_token_drop = torch.zeros(batch_size, dtype=dtype)
        timestep = torch.randint(0, 1000, (batch_size,))
        target_size = torch.tensor([[512, 512]] * batch_size, dtype=dtype)
        crop_coords = torch.tensor([[0, 0]] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "prior_token_id": prior_token_id,
            "prior_token_drop": prior_token_drop,
            "timestep": timestep,
            "target_size": target_size,
            "crop_coords": crop_coords,
            "return_dict": False,
        }
