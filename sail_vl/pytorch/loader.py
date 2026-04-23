# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAIL-VL model loader implementation for multimodal visual question answering.
"""

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from typing import Optional

from ...tools.utils import get_file
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Variants that use the HF-native format (AutoProcessor + apply_chat_template).
_HF_NATIVE_VARIANTS = {"2_2B"}


def build_transform(input_size):
    """Build image transform pipeline for SAIL-VL."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    """Split image into tiles using dynamic resolution."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_path, input_size=448, max_num=12):
    """Load and preprocess an image for SAIL-VL."""
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class ModelVariant(StrEnum):
    """Available SAIL-VL model variants."""

    SAIL_VL_1D6_8B = "1d6_8B"
    SAIL_VL2_2B = "2_2B"


class ModelLoader(ForgeModel):
    """SAIL-VL model loader implementation for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.SAIL_VL_1D6_8B: ModelConfig(
            pretrained_model_name="BytedanceDouyinContent/SAIL-VL-1d6-8B",
        ),
        ModelVariant.SAIL_VL2_2B: ModelConfig(
            pretrained_model_name="BytedanceDouyinContent/SAIL-VL2-2B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAIL_VL_1D6_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
        self.model = None

    @property
    def _is_hf_native(self):
        return self._variant.value in _HF_NATIVE_VARIANTS

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SAIL-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        import sys
        import transformers.processing_utils as _pu

        if not hasattr(_pu, "_validate_images_text_input_order"):
            _pu._validate_images_text_input_order = lambda images, text: (images, text)
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        # SAILVLConfig default llm_config uses Qwen2ForCausalLM which the cached code
        # does not handle; setting has_no_defaults_at_init skips the no-arg instantiation
        # in to_diff_dict() that would otherwise raise ValueError.
        for mod_name, module in list(sys.modules.items()):
            if "configuration_sailvl" in mod_name and hasattr(module, "SAILVLConfig"):
                module.SAILVLConfig.has_no_defaults_at_init = True
                break
        return self.processor

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._is_hf_native:
            if self.processor is None:
                self._load_processor()
        elif self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model

        return model

    def _load_inputs_hf_native(self, dtype_override=None, batch_size=1):
        """Load inputs for HF-native variants using AutoProcessor."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_file)},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        for key in inputs:
            if torch.is_tensor(inputs[key]) and batch_size > 1:
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def _load_inputs_custom(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        pixel_values = load_image(image_file, max_num=12)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        num_patches = pixel_values.shape[0]

        # Set up image context token id
        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if self.model is not None:
            self.model.img_context_token_id = img_context_token_id

        # Build prompt
        question = "<image>\nWhat is shown in this image?"
        messages = [{"role": "user", "content": question}]
        query = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Replace <image> placeholder with the correct number of image tokens
        num_image_token = self.model.num_image_token if self.model is not None else 256
        image_tokens = (
            "<img>" + "<IMG_CONTEXT>" * num_image_token * num_patches + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)

        model_inputs = self.tokenizer(
            query, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
            image_flags = image_flags.repeat_interleave(batch_size, dim=0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
            "use_cache": False,
        }

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self._is_hf_native:
            return self._load_inputs_hf_native(
                dtype_override=dtype_override, batch_size=batch_size
            )
        return self._load_inputs_custom(
            dtype_override=dtype_override, batch_size=batch_size
        )

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self._is_hf_native:
            if self.processor is None:
                self._load_processor()
            tokenizer = self.processor.tokenizer
        else:
            if self.tokenizer is None:
                self._load_tokenizer()
            tokenizer = self.tokenizer

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return tokenizer.decode(next_token_id)
