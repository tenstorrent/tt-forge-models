# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cambrian model loader implementation for multimodal visual question answering.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import Optional

# Register cambrian_qwen architecture and patch transformers compat issues
from .src.model_utils import CambrianQwenForCausalLM  # noqa: F401

from ...tools.utils import get_file, cast_input_to_type
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


class ModelVariant(StrEnum):
    """Available Cambrian model variants."""

    CAMBRIAN_S_7B_LFP = "S_7B_LFP"


class ModelLoader(ForgeModel):
    """Cambrian model loader implementation for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.CAMBRIAN_S_7B_LFP: ModelConfig(
            pretrained_model_name="nyu-visionx/Cambrian-S-7B-LFP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CAMBRIAN_S_7B_LFP

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Cambrian",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        question = "What is shown in this image?"
        # Use string content with <image> placeholder (Qwen2 tokenizer does
        # not support multimodal content dicts in apply_chat_template).
        messages = [
            {
                "role": "user",
                "content": f"<image>\n{question}",
            }
        ]

        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt")

        # Preprocess image through the vision tower's image processor.
        # The vision tower uses SigLIP which expects 384x384 images.
        from torchvision import transforms

        image_transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        pixel_values = image_transform(image.convert("RGB")).unsqueeze(0)
        inputs["images"] = pixel_values

        if dtype_override is not None:
            inputs["images"] = cast_input_to_type(inputs["images"], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
