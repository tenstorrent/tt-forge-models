# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit Middle-Finger LoRA model loader implementation.

Loads the Qwen-Image-Edit base diffusion pipeline and applies the
drbaph/Qwen-Image-Edit-Middle-Finger-LoRA adapter weights for
image-to-image editing that adds middle-finger gestures to subjects.

Available variants:
- MIDDLE_FINGER_V1: Middle-Finger LoRA v1.0 on Qwen-Image-Edit
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPipeline

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

BASE_MODEL = "Qwen/Qwen-Image-Edit"
LORA_REPO = "drbaph/Qwen-Image-Edit-Middle-Finger-LoRA"
LORA_WEIGHT_NAME = "qwen_image_edit_middle-finger_lora_v1.0.safetensors"

# VAE compresses by 8x spatially; latents are packed into 2x2 patches.
_VAE_SCALE_FACTOR = 8


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit Middle-Finger LoRA variants."""

    MIDDLE_FINGER_V1 = "MiddleFinger_v1"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit Middle-Finger LoRA model loader."""

    _VARIANTS = {
        ModelVariant.MIDDLE_FINGER_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MIDDLE_FINGER_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_MIDDLE_FINGER_LORA",
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
        """Load the Qwen-Image-Edit pipeline with Middle-Finger LoRA weights.

        Returns the underlying transformer (torch.nn.Module) so the test
        harness can call it directly with tensor inputs.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline.transformer

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare synthetic tensor inputs for the QwenImageTransformer2DModel.

        The pipeline concatenates noisy target latents with encoded image latents
        before passing them to the transformer.  We replicate that layout here
        with random tensors so no actual image or prompt encoding is required.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        transformer = self.pipeline.transformer
        dtype = dtype_override if dtype_override is not None else transformer.dtype
        config = transformer.config

        # in_channels=64 → 16 latent channels per spatial position; packed 2×2 → ×4
        num_channels_latents = config.in_channels // 4

        height, width = 256, 256
        # prepare_latents rounds to nearest even multiple of vae_scale_factor*2
        h_lat = 2 * (height // (_VAE_SCALE_FACTOR * 2))  # 32
        w_lat = 2 * (width // (_VAE_SCALE_FACTOR * 2))  # 32
        # _pack_latents folds each 2×2 spatial block into a single token
        h_packed = h_lat // 2  # 16
        w_packed = w_lat // 2  # 16
        seq_len = h_packed * w_packed  # 256

        # target noisy latents + source image latents concatenated on seq dim
        latents = torch.randn(1, seq_len, num_channels_latents * 4, dtype=dtype)
        image_latents = torch.randn(1, seq_len, num_channels_latents * 4, dtype=dtype)
        hidden_states = torch.cat([latents, image_latents], dim=1)

        # timestep already divided by 1000 as the pipeline does before passing
        timestep = torch.tensor([0.5], dtype=dtype)

        # synthetic text embeddings; joint_attention_dim=3584 for this model
        text_seq_len = 64
        encoder_hidden_states = torch.randn(
            1, text_seq_len, config.joint_attention_dim, dtype=dtype
        )

        # img_shapes encodes the (1, h, w) spatial layout for each sample;
        # two entries per sample: target shape then source-image shape
        img_shapes = [
            [
                (1, h_packed, w_packed),
                (1, h_packed, w_packed),
            ]
        ]

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": None,
            "encoder_hidden_states_mask": None,
            "encoder_hidden_states": encoder_hidden_states,
            "img_shapes": img_shapes,
            "attention_kwargs": None,
            "return_dict": False,
        }

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        if isinstance(fwd_output, (tuple, list)):
            return fwd_output[0]
        return fwd_output
