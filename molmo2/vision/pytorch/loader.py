# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 vision-tower loader (image encoder only).

Molmo2-8B's vision tower is a SigLIP-style ViT (``Molmo2VisionTransformer``):
a linear patch embedding over pre-extracted 14x14 RGB patches, learned
positional embeddings, and 27 pre-norm attention blocks (hidden 1152, 16 heads,
head dim 72, gelu-tanh MLP). Unlike CLIP/SigLIP towers in other VLMs it uses a
``nn.Linear`` patch embed (the image processor pre-extracts flattened patches),
so there is no Conv2d to legalize.

This loader wraps just the ViT and returns its final hidden state. The
downstream image-pooling adapter (multi-head attention pooling + projector) is
intentionally excluded here: it relies on data-dependent gather / boolean-mask
indexing that the static-shape device path cannot express, so it is brought up
(or root-caused) separately. The custom modeling code targets
transformers==4.57.1 (see requirements.txt).
"""

from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import cast_input_to_type

_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"


class _VisionTowerWrapper(nn.Module):
    """Runs the Molmo2 ViT and returns the final-layer hidden state tensor."""

    def __init__(self, image_vit):
        super().__init__()
        self.image_vit = image_vit

    def forward(self, pixel_values):
        # image_vit returns a list of per-block hidden states; take the last.
        hidden_states = self.image_vit(pixel_values)
        return hidden_states[-1]


class ModelVariant(StrEnum):
    """Available Molmo2 vision-tower variants."""

    MOLMO2_8B = "8B"


class ModelLoader(ForgeModel):
    """Molmo2 vision-tower (image encoder) loader."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: ModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=_REVISION,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full model and return a wrapper around its ViT tower."""
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.float32
        full = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=_REVISION,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        full.eval()
        self.config = full.config

        image_vit = full.model.vision_backbone.image_vit
        wrapper = _VisionTowerWrapper(image_vit).eval()
        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)

        if self.processor is None:
            self._load_processor()
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return the ViT input tensor (pre-extracted image patches)."""
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.sample_text},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # pixel_values: (num_crops, num_patch, n_pixels) e.g. (9, 729, 588)
        pixel_values = inputs["pixel_values"]
        if batch_size != 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return {"pixel_values": cast_input_to_type(pixel_values, dtype_override)}

    def load_config(self):
        if self.config is None:
            self.load_model()
        return self.config
