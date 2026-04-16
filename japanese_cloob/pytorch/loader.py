# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Japanese CLOOB model loader implementation for image-text similarity.

Loads the rinna/japanese-cloob-vit-b-16 model by constructing it from
BERT (text encoder) + ViT (vision encoder) + projection layers and
loading weights directly from HuggingFace, since the original
japanese_clip package is not available on PyPI.
"""
import json

import torch
import torch.nn as nn
from typing import Optional
from transformers import (
    BertModel,
    BertConfig,
    ViTModel,
    ViTConfig,
    AutoTokenizer,
    CLIPImageProcessor,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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
from datasets import load_dataset


class CLOOBModel(nn.Module):
    """CLOOB model combining a ViT vision encoder and BERT text encoder with projections."""

    def __init__(self, vision_model, text_model, visual_projection, text_projection):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.visual_projection = visual_projection
        self.text_projection = text_projection

    def get_image_features(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state[:, 0]
        return self.visual_projection(pooled)

    def get_text_features(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.text_projection(pooled)


def _load_cloob_model(model_name):
    """Load a CLOOB model from HuggingFace by constructing from components."""
    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    text_config = BertConfig(**config["text_config"])
    vision_config = ViTConfig(**config["vision_config"])
    projection_dim = config["projection_dim"]

    text_model = BertModel(text_config, add_pooling_layer=False)
    vision_model = ViTModel(vision_config, add_pooling_layer=False)
    text_projection = nn.Linear(text_config.hidden_size, projection_dim, bias=False)
    visual_projection = nn.Linear(vision_config.hidden_size, projection_dim, bias=False)

    model = CLOOBModel(vision_model, text_model, visual_projection, text_projection)

    weights_path = hf_hub_download(model_name, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)

    return model


class JapaneseCLOOBWrapper(nn.Module):
    """Wrapper to combine image and text feature extraction into a single forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, input_ids, attention_mask):
        image_features = self.model.get_image_features(image)
        text_features = self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return image_features, text_features


class ModelVariant(StrEnum):
    """Available Japanese CLOOB model variants."""

    VIT_B_16 = "ViT_B_16"


class ModelLoader(ForgeModel):
    """Japanese CLOOB model loader implementation for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.VIT_B_16: ModelConfig(
            pretrained_model_name="rinna/japanese-cloob-vit-b-16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_B_16

    _TOKENIZER_NAME = "rinna/japanese-roberta-base"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.image_processor = None
        self.tokenizer = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Japanese_CLOOB",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Japanese CLOOB model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Japanese CLOOB model instance.
        """
        model = _load_cloob_model(self._variant_config.pretrained_model_name)

        wrapper = JapaneseCLOOBWrapper(model)

        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)

        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Japanese CLOOB model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.image_processor is None:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch16"
            )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Preprocess the image
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values

        # Define Japanese text prompts (cat, dog, elephant)
        self.text_prompts = ["猫", "犬", "象"]

        # Tokenize text
        text_inputs = self.tokenizer(
            self.text_prompts,
            return_tensors="pt",
            padding=True,
            max_length=77,
            truncation=True,
        )

        inputs = {
            "image": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["image"] = inputs["image"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process model outputs to extract similarity scores.

        Args:
            outputs: Raw model output (image_features, text_features)
        """
        if self.text_prompts is None:
            self.text_prompts = ["猫", "犬", "象"]

        image_features, text_features = outputs[0], outputs[1]

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", similarity[0, i].item())
