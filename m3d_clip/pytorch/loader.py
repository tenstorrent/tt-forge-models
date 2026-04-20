# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GoodBaiBai88/M3D-CLIP model loader for 3D medical image-text similarity.

M3D-CLIP aligns volumetric medical imagery (e.g. CT/MRI) with text via a 3D
Vision Transformer vision encoder and a BERT language encoder, trained
contrastively on the M3D-Cap dataset.
"""
import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available M3D-CLIP model variants."""

    M3D_CLIP = "M3D-CLIP"


class ModelLoader(ForgeModel):
    """M3D-CLIP model loader for 3D medical image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.M3D_CLIP: ModelConfig(
            pretrained_model_name="GoodBaiBai88/M3D-CLIP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.M3D_CLIP

    # Fixed 3D input shape required by the pretrained model
    # (channels, depth, height, width)
    IMAGE_SHAPE = (1, 32, 256, 256)
    MAX_TEXT_LEN = 128

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="M3D-CLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                model_max_length=self.MAX_TEXT_LEN,
                padding_side="right",
                use_fast=False,
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the M3D-CLIP model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The M3D-CLIP model instance.
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
        """Load and return sample inputs for the M3D-CLIP model.

        The vision encoder expects volumetric input of shape (B, 1, 32, 256, 256)
        normalized to [0, 1]. Since the pretrained preprocessor only ships as a
        loose preprocessing recipe (normalize a `.npy` volume), we generate a
        deterministic random volume here for graph capture / inference tests.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing images, input_ids, attention_mask, labels.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        self.text_prompts = [
            "A CT scan showing normal lung parenchyma without abnormalities.",
        ] * batch_size

        text_tensor = self.tokenizer(
            self.text_prompts,
            max_length=self.MAX_TEXT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        generator = torch.Generator().manual_seed(0)
        images = torch.rand(
            (batch_size, *self.IMAGE_SHAPE),
            generator=generator,
        )

        if dtype_override is not None:
            images = images.to(dtype_override)

        labels = torch.arange(batch_size, dtype=torch.long)

        return {
            "images": images,
            "input_ids": text_tensor["input_ids"],
            "attention_mask": text_tensor["attention_mask"],
            "labels": labels,
        }

    def post_process(self, outputs):
        """Print the contrastive logits produced by the model.

        Args:
            outputs: Raw model output (dict with ``loss`` and ``logits``).
        """
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = outputs[1] if len(outputs) > 1 else outputs[0]

        print("Image-text contrastive logits:", logits)

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass (dict of tensors).

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass.
        """
        if isinstance(fwd_output, dict):
            tensors = [t.flatten() for t in fwd_output.values() if torch.is_tensor(t)]
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
