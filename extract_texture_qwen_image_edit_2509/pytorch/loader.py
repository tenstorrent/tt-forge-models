# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Extract-Texture Qwen-Image-Edit-2509 LoRA model loader implementation.

Loads the Qwen-Image-Edit-2509 base pipeline and applies the
tarn59/extract_texture_qwen_image_edit_2509 LoRA weights for extracting
clean, tileable texture images from objects in input images.

Available variants:
- EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509: Texture extraction LoRA
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline  # type: ignore[import]

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "tarn59/extract_texture_qwen_image_edit_2509"


class ModelVariant(StrEnum):
    """Available Extract-Texture Qwen-Image-Edit-2509 variants."""

    EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509 = "Extract_Texture_Qwen_Image_Edit_2509"


class ModelLoader(ForgeModel):
    """Extract-Texture Qwen-Image-Edit-2509 LoRA model loader."""

    _VARIANTS = {
        ModelVariant.EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509",
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
        """Load the Qwen-Image-Edit-2509 pipeline with Extract-Texture LoRA.

        Returns:
            QwenImageTransformer2DModel (transformer sub-module).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline.transformer

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare synthetic tensor inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() for
        image editing (output latents + one input image latents concatenated).
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        transformer = self.pipeline.transformer
        batch_size = kwargs.get("batch_size", 1)

        in_channels = transformer.config.in_channels  # 64
        joint_attention_dim = transformer.config.joint_attention_dim  # 3584
        patch_size = transformer.config.patch_size  # 2

        vae_scale_factor = 8
        # Synthetic 512x512 image through VAE then patch embedding
        image_size = 512
        lat = image_size // vae_scale_factor  # 64
        lat_packed = lat // patch_size  # 32
        tokens_per_image = lat_packed * lat_packed  # 1024
        # image editing: output latents + 1 input image latents
        num_input_images = 1
        total_tokens = tokens_per_image * (1 + num_input_images)

        txt_seq_len = 32

        hidden_states = torch.randn(batch_size, total_tokens, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, joint_attention_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([0.5] * batch_size, dtype=dtype)
        img_shapes = [
            [(1, lat_packed, lat_packed)] * (1 + num_input_images)
        ] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack transformer output tuple to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
