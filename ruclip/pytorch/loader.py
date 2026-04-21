# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RuCLIP model loader implementation for Russian image-text similarity.

Reference: https://huggingface.co/ai-forever/ruclip-vit-base-patch32-384
"""
import torch
from datasets import load_dataset
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


class ModelVariant(StrEnum):
    """Available RuCLIP model variants for Russian image-text similarity."""

    VIT_BASE_PATCH32_384 = "ViT_Base_Patch32_384"


class ModelLoader(ForgeModel):
    """RuCLIP model loader implementation for Russian image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.VIT_BASE_PATCH32_384: ModelConfig(
            pretrained_model_name="ai-forever/ruclip-vit-base-patch32-384",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_BASE_PATCH32_384

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="RuCLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _download_repo(self):
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=self._variant_config.pretrained_model_name)

    def _load_processor(self):
        from ruclip.processor import RuCLIPProcessor

        self.processor = RuCLIPProcessor.from_pretrained(self._download_repo())
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RuCLIP model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The RuCLIP model instance for image-text similarity.
        """
        from ruclip.model import CLIP

        model = CLIP.from_pretrained(self._download_repo())
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RuCLIP model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors containing input_ids and pixel_values.
        """
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Russian text prompts for image-text similarity
        self.text_prompts = ["кошка", "собака", "слон"]

        processed = self.processor(text=self.text_prompts, images=[image])
        inputs = {
            "input_ids": processed["input_ids"],
            "pixel_values": processed["pixel_values"],
        }

        # Replicate pixel_values to match the number of text prompts
        num_texts = len(self.text_prompts)
        inputs["pixel_values"] = inputs["pixel_values"].expand(num_texts, -1, -1, -1)

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process RuCLIP model outputs to extract similarity scores.

        Args:
            outputs: Raw model output (logits_per_image, logits_per_text).
        """
        if self.text_prompts is None:
            self.text_prompts = ["кошка", "собака", "слон"]

        logits_per_image = outputs[0]
        probs = logits_per_image.softmax(dim=-1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        The RuCLIP model returns a tuple of (logits_per_image, logits_per_text).

        Args:
            fwd_output: Output from the model's forward pass (tuple).

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass.
        """
        if isinstance(fwd_output, tuple):
            tensors = [item.flatten() for item in fwd_output if torch.is_tensor(item)]
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
