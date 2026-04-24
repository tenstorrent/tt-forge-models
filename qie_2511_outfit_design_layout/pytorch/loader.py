# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QIE-2511-Outfit-Design-Layout LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2511 base diffusion pipeline and applies
the prithivMLmods/QIE-2511-Outfit-Design-Layout LoRA adapter for adding
outfit elements or design layouts into marked image regions.

Available variants:
- QIE_2511_OUTFIT_DESIGN_LAYOUT: Outfit Design Layout LoRA (bf16)
"""

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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO_ID = "prithivMLmods/QIE-2511-Outfit-Design-Layout"


class ModelVariant(StrEnum):
    """Available QIE-2511-Outfit-Design-Layout model variants."""

    QIE_2511_OUTFIT_DESIGN_LAYOUT = "Outfit_Design_Layout"


class ModelLoader(ForgeModel):
    """QIE-2511-Outfit-Design-Layout LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QIE_2511_OUTFIT_DESIGN_LAYOUT: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QIE_2511_OUTFIT_DESIGN_LAYOUT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QIE_2511_Outfit_Design_Layout",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the Qwen-Image-Edit-2511 transformer with Outfit Design Layout LoRA merged.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            pipe = DiffusionPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            pipe.load_lora_weights(LORA_REPO_ID)
            pipe.fuse_lora()
            self._transformer = pipe.transformer
            self._transformer.eval()
            del pipe
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)

        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64, joint_attention_dim=3584
        img_dim = 64
        text_dim = 3584
        txt_seq_len = 32

        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
