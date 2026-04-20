# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
calcuis/gguf-node AuraFlow v0.3 GGUF model loader.

Loads the AuraFlow v0.3 transformer packaged as a GGUF checkpoint inside the
calcuis/gguf-node collection. The transformer architecture/config is taken
from fal/AuraFlow-v0.3 while the weights come from the quantized GGUF file.

Repository: https://huggingface.co/calcuis/gguf-node
"""

from typing import Optional

import torch
from diffusers import AuraFlowTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

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

REPO_ID = "calcuis/gguf-node"
TRANSFORMER_CONFIG = "fal/AuraFlow-v0.3"
TRANSFORMER_SUBFOLDER = "transformer"


class ModelVariant(StrEnum):
    """Available calcuis/gguf-node AuraFlow variants."""

    AURA_FLOW_0_3_Q4_0 = "aura_flow_0.3_q4_0"


class ModelLoader(ForgeModel):
    """calcuis/gguf-node AuraFlow v0.3 loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.AURA_FLOW_0_3_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.AURA_FLOW_0_3_Q4_0: "aura_flow_0.3-q4_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.AURA_FLOW_0_3_Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GGUF_Node_AuraFlow",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.transformer is None:
            gguf_path = hf_hub_download(
                repo_id=self._variant_config.pretrained_model_name,
                filename=self._GGUF_FILES[self._variant],
            )
            self.transformer = AuraFlowTransformer2DModel.from_single_file(
                gguf_path,
                config=TRANSFORMER_CONFIG,
                subfolder=TRANSFORMER_SUBFOLDER,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            self.transformer.eval()
        elif dtype_override is not None:
            self.transformer = self.transformer.to(dtype=dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.transformer.config

        # Latent image: (B, in_channels, H, W)
        height = 128
        width = 128
        hidden_states = torch.randn(
            batch_size, config.in_channels, height, width, dtype=dtype
        )

        # Text encoder hidden states from Pile-T5-XL: (B, seq_len, joint_attention_dim)
        max_sequence_length = 256
        encoder_hidden_states = torch.randn(
            batch_size,
            max_sequence_length,
            config.joint_attention_dim,
            dtype=dtype,
        )

        # Timestep: (B,)
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }

        return inputs
