# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoLLaMA3-7B model loader implementation for multimodal video understanding.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available VideoLLaMA3 model variants."""

    BASE_7B = "base_7b"


class ModelLoader(ForgeModel):
    """VideoLLaMA3-7B model loader for multimodal video understanding."""

    _VARIANTS = {
        ModelVariant.BASE_7B: ModelConfig(
            pretrained_model_name="DAMO-NLP-SG/VideoLLaMA3-7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoLLaMA3 model loader."""
        super().__init__(variant)
        self.model_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoLLaMA3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoLLaMA3 model instance."""
        model_name = self._variant_config.pretrained_model_name
        kwargs.setdefault("trust_remote_code", True)
        model = AutoModelForCausalLM.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        self.model_config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return synthetic input tensors for VideoLLaMA3."""
        model_name = self._variant_config.pretrained_model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        image_token_index = getattr(self.model_config, "image_token_index", 151652)
        vision_cfg = getattr(self.model_config, "vision_encoder_config", None)
        patch_size = getattr(vision_cfg, "patch_size", 16) if vision_cfg else 16
        num_channels = getattr(vision_cfg, "num_channels", 3) if vision_cfg else 3

        prompt = "Describe what is happening in this video."
        tokens = tokenizer(prompt, return_tensors="pt")
        text_ids = tokens["input_ids"]

        num_frames = 2
        grid_h = 2
        grid_w = 2
        merge_size = 1
        num_patches_per_frame = grid_h * grid_w
        total_patches = num_frames * num_patches_per_frame
        num_image_tokens = total_patches

        image_token_ids = torch.full(
            (1, num_image_tokens), image_token_index, dtype=torch.long
        )
        input_ids = torch.cat([text_ids, image_token_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        pixel_dim = num_channels * patch_size * patch_size
        pixel_values = torch.randn(total_patches, pixel_dim)

        grid_sizes = torch.tensor([[num_frames, grid_h, grid_w]], dtype=torch.long)
        merge_sizes = torch.tensor([merge_size], dtype=torch.long)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "grid_sizes": grid_sizes,
            "merge_sizes": merge_sizes,
            "modals": ["video"],
        }

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()
            }

        return inputs
