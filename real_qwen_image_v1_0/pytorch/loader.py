# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Real-Qwen-Image-v1.0 GGUF model loader implementation.

Loads the GGUF-quantized diffusion transformer from
wikeeyang/Real-Qwen-Image-v1.0, a Qwen-Image fine-tune that enhances
clarity and realism for text-to-image generation. The repository only
ships single-file checkpoints, so the diffusers config is sourced from
the upstream Qwen/Qwen-Image repository.

Available variants:
- Q4_K_M: 4-bit Q4_K_M GGUF quantization
- Q8_0: 8-bit Q8_0 GGUF quantization
"""

from typing import Any, Optional

import torch
from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
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

REPO_ID = "wikeeyang/Real-Qwen-Image-v1.0"
CONFIG_REPO = "Qwen/Qwen-Image"


class ModelVariant(StrEnum):
    """Available Real-Qwen-Image-v1.0 GGUF model variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_K_M: "Real-Qwen-Image-V1-Q4_K_M.gguf",
    ModelVariant.Q8_0: "Real-Qwen-Image-V1-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Real-Qwen-Image-v1.0 GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Real_Qwen_Image_v1_0_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Real-Qwen-Image-v1.0 GGUF diffusion transformer.

        Returns:
            QwenImageTransformer2DModel instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        gguf_file = _GGUF_FILES[self._variant]
        repo_id = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(repo_id=repo_id, filename=gguf_file)
        self._transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            quantization_config=quantization_config,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=compute_dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From Qwen-Image transformer config: in_channels=64
        img_dim = 64
        # joint_attention_dim from Qwen-Image config = 3584
        text_dim = 3584
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
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
