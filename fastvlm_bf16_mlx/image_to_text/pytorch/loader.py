# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FastVLM bf16 MLX model loader implementation for image-to-text tasks.
"""
import re

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file

IMAGE_TOKEN_INDEX = -200


class ModelVariant(StrEnum):
    """Available FastVLM bf16 MLX model variants for image-to-text tasks."""

    FASTVLM_0_5B_BF16 = "fastvlm_0_5b_bf16"


class ModelLoader(ForgeModel):
    """FastVLM bf16 MLX model loader implementation for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.FASTVLM_0_5B_BF16: LLMModelConfig(
            pretrained_model_name="mlx-community/FastVLM-0.5B-bf16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FASTVLM_0_5B_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="fastvlm_bf16_mlx",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    @staticmethod
    def _remap_checkpoint_keys(state_dict: dict) -> dict:
        """Remap mlx-community checkpoint keys to match llava_qwen.py parameter names.

        The mlx-community/FastVLM-0.5B-bf16 checkpoint was saved from an MLX-converted
        model with a different attribute layout and tensor layout than llava_qwen.py expects:

        Key remapping:
          language_model.model.X                          ->  model.X
          language_model.lm_head.X                        ->  lm_head.X
          mm_projector.X                                  ->  model.mm_projector.X
          vision_tower.vision_model.patch_embed.blocks.N.X
                                                          ->  model.vision_tower.vision_tower.model.patch_embed.N.X
          vision_tower.vision_model.X                     ->  model.vision_tower.vision_tower.model.X

        Tensor layout remapping for vision tower weights (MLX NHWC → PyTorch NCHW):
          4D tensors: (out_c, kH, kW, in_c) -> permute(0, 3, 1, 2) -> (out_c, in_c, kH, kW)
          3D tensors: (1, 1, C)             -> permute(2, 0, 1)     -> (C, 1, 1)
        """
        remapped = {}
        for key, val in state_dict.items():
            if key.startswith("language_model.model."):
                new_key = "model." + key[len("language_model.model."):]
            elif key.startswith("language_model.lm_head."):
                new_key = "lm_head." + key[len("language_model.lm_head."):]
            elif key.startswith("mm_projector."):
                new_key = "model." + key
            elif key.startswith("vision_tower.vision_model."):
                rest = key[len("vision_tower.vision_model."):]
                # patch_embed.blocks.N.X  ->  patch_embed.N.X
                rest = re.sub(r"^patch_embed\.blocks\.(\d+)\.", r"patch_embed.\1.", rest)
                new_key = "model.vision_tower.vision_tower.model." + rest
                # Permute MLX-format weight tensors to PyTorch layout.
                if val.dim() == 4:
                    val = val.permute(0, 3, 1, 2).contiguous()
                elif val.dim() == 3:
                    val = val.permute(2, 0, 1).contiguous()
            else:
                new_key = key
            remapped[new_key] = val
        return remapped

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FastVLM bf16 MLX model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The FastVLM bf16 MLX model instance for image-to-text tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Build model from config (avoids the broken from_pretrained weight load).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=True)

        # Load and remap the safetensors checkpoint.
        ckpt_path = hf_hub_download(pretrained_model_name, "model.safetensors")
        raw_sd = load_file(ckpt_path, device="cpu")
        remapped_sd = self._remap_checkpoint_keys(raw_sd)
        missing, unexpected = model.load_state_dict(remapped_sd, strict=False)

        # Tie weights (lm_head <-> embed_tokens) after loading.
        model.tie_weights()

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FastVLM bf16 MLX model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [
            {
                "role": "user",
                "content": "<image>\nDescribe this image in detail.",
            }
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        pre, post = rendered.split("<image>", 1)
        pre_ids = self.tokenizer(
            pre, return_tensors="pt", add_special_tokens=False
        ).input_ids
        post_ids = self.tokenizer(
            post, return_tensors="pt", add_special_tokens=False
        ).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        vision_tower = self.model.get_vision_tower()
        pixel_values = vision_tower.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": pixel_values,
        }
