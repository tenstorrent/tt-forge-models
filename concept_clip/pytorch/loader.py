# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ConceptCLIP model loader implementation for biomedical image-text similarity.
"""
import torch
from transformers import AutoModel, AutoProcessor
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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available ConceptCLIP model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """ConceptCLIP model loader for biomedical image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="JerrryNie/ConceptCLIP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CONCEPTCLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ConceptCLIP model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ConceptCLIP model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ConceptCLIP model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors for the model.
        """
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        labels = ["chest X-ray", "brain MRI", "skin lesion"]
        self.text_prompts = [f"a medical image of {label}" for label in labels]

        inputs = self.processor(
            images=image,
            text=self.text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process ConceptCLIP model outputs to extract similarity scores.

        Args:
            outputs: Raw model output containing image_features, text_features, logit_scale.
        """
        if self.text_prompts is None:
            labels = ["chest X-ray", "brain MRI", "skin lesion"]
            self.text_prompts = [f"a medical image of {label}" for label in labels]

        image_features = outputs["image_features"]
        text_features = outputs["text_features"]
        logit_scale = outputs["logit_scale"]

        probs = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)[0]

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass.

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass.
        """
        if isinstance(fwd_output, dict):
            tensors = []
            for value in fwd_output.values():
                if isinstance(value, torch.Tensor):
                    tensors.append(value.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
                elif hasattr(item, "last_hidden_state"):
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
