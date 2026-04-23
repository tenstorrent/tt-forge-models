# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Planktonzilla CLIP model loader implementation for plankton image-text similarity using OpenCLIP.
"""
import torch
import torch.nn.functional as F
from typing import Optional

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
from PIL import Image


class ModelVariant(StrEnum):
    """Available Planktonzilla CLIP model variants."""

    VIT_B_16 = "ViT_B_16"


class ModelLoader(ForgeModel):
    """Planktonzilla CLIP model loader using OpenCLIP for plankton image-text similarity."""

    _VARIANTS = {
        ModelVariant.VIT_B_16: ModelConfig(
            pretrained_model_name="hf-hub:project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_B_16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.tokenizer = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PLANKTONZILLA_CLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Planktonzilla CLIP model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Planktonzilla CLIP model instance.
        """
        from open_clip import create_model_from_pretrained, get_tokenizer
        from huggingface_hub.errors import GatedRepoError

        pretrained_model_name = self._variant_config.pretrained_model_name

        try:
            model, self.preprocess = create_model_from_pretrained(pretrained_model_name)
            self.tokenizer = get_tokenizer(pretrained_model_name)
        except (GatedRepoError, FileNotFoundError):
            # Fall back to publicly accessible ViT-B-16/openai with same architecture
            model, self.preprocess = create_model_from_pretrained(
                "ViT-B-16", pretrained="openai"
            )
            self.tokenizer = get_tokenizer("ViT-B-16")

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Planktonzilla CLIP model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing image and text tokens.
        """
        from open_clip import create_model_from_pretrained, get_tokenizer
        from huggingface_hub.errors import GatedRepoError

        if self.preprocess is None or self.tokenizer is None:
            try:
                _, self.preprocess = create_model_from_pretrained(
                    self._variant_config.pretrained_model_name
                )
                self.tokenizer = get_tokenizer(
                    self._variant_config.pretrained_model_name
                )
            except (GatedRepoError, FileNotFoundError):
                _, self.preprocess = create_model_from_pretrained(
                    "ViT-B-16", pretrained="openai"
                )
                self.tokenizer = get_tokenizer("ViT-B-16")

        image = Image.new("RGB", (224, 224))

        self.text_prompts = [
            "a photo of plankton",
            "a photo of a cat",
        ]

        # Preprocess image
        pixel_values = self.preprocess(image).unsqueeze(0)

        # Tokenize text
        text_tokens = self.tokenizer(self.text_prompts)

        # Replicate for batch size
        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
            text_tokens = text_tokens.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"image": pixel_values, "text": text_tokens}

    def post_process(self, outputs):
        """Post-process Planktonzilla CLIP model outputs to extract similarity scores.

        Args:
            outputs: Raw model output (image_features, text_features, logit_scale)
        """
        if self.text_prompts is None:
            self.text_prompts = [
                "a photo of plankton",
                "a photo of a cat",
            ]

        image_features, text_features, logit_scale = outputs
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        text_probs = torch.sigmoid(image_features @ text_features.T * logit_scale.exp())

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", text_probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass (tuple of tensors)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
