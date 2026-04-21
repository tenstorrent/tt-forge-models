# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MEXMA-SigLIP2 model loader implementation for multilingual image-text similarity.
"""
import torch
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available MEXMA-SigLIP2 model variants."""

    MEXMA_SIGLIP2 = "visheratin/mexma-siglip2"


class ModelLoader(ForgeModel):
    """MEXMA-SigLIP2 model loader for multilingual zero-shot image-text similarity."""

    _VARIANTS = {
        ModelVariant.MEXMA_SIGLIP2: ModelConfig(
            pretrained_model_name="visheratin/mexma-siglip2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEXMA_SIGLIP2

    text_prompts = ["a photo of a cat", "a photo of a dog"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MEXMA-SigLIP2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def _load_processor(self):
        from transformers import AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MEXMA-SigLIP2 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The MEXMA-SigLIP2 model instance.
        """
        from transformers import AutoModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the MEXMA-SigLIP2 model.

        Args:
            dtype_override: Optional torch.dtype to override the image input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors with image pixel values, text input ids, and attention mask.
        """
        from datasets import load_dataset

        if self.tokenizer is None:
            self._load_tokenizer()
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        image_inputs = self.processor(images=image, return_tensors="pt")["pixel_values"]

        text_inputs = self.tokenizer(
            self.text_prompts, return_tensors="pt", padding=True
        )

        if batch_size > 1:
            image_inputs = image_inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            image_inputs = image_inputs.to(dtype_override)

        return {
            "image_inputs": image_inputs,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    def decode_output(self, outputs):
        """Decode MEXMA-SigLIP2 outputs into per-prompt similarity probabilities.

        Args:
            outputs: Forward output dict with image_features, text_features, logit_scale, logit_bias.

        Returns:
            torch.Tensor: Softmax probabilities over text prompts per image.
        """
        if isinstance(outputs, dict):
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logit_scale = outputs["logit_scale"]
            logit_bias = outputs.get("logit_bias", 0.0)
        else:
            image_features, text_features, logit_scale = outputs[:3]
            logit_bias = outputs[3] if len(outputs) > 3 else 0.0

        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        image_logits = image_features @ text_features.T * logit_scale.exp() + logit_bias
        return image_logits.softmax(dim=-1)

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output into a single differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass (dict or tuple of tensors).

        Returns:
            torch.Tensor: Concatenated flattened tensor outputs.
        """
        if isinstance(fwd_output, dict):
            items = fwd_output.values()
        elif isinstance(fwd_output, (tuple, list)):
            items = fwd_output
        else:
            return fwd_output

        tensors = [item.flatten() for item in items if isinstance(item, torch.Tensor)]
        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
