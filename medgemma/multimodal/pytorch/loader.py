# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedGemma model loader implementation for multimodal modeling.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type, get_file
from PIL import Image


class ModelVariant(StrEnum):
    """Available MedGemma multimodal model variants."""

    MEDGEMMA_4B_PT = "google/medgemma-4b-pt"
    UNSLOTH_MEDGEMMA_4B_IT_BNB_4BIT = "unsloth/medgemma-4b-it-unsloth-bnb-4bit"


class ModelLoader(ForgeModel):
    """MedGemma model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.MEDGEMMA_4B_PT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MEDGEMMA_4B_PT),
        ),
        ModelVariant.UNSLOTH_MEDGEMMA_4B_IT_BNB_4BIT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.UNSLOTH_MEDGEMMA_4B_IT_BNB_4BIT),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEDGEMMA_4B_PT

    sample_text = "Describe this medical image."
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="medgemma_multimodal",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        kwargs = {"use_fast": False}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name, **kwargs)

        return self.processor

    @staticmethod
    def _dequantize_bnb_linear4bit(model):
        """Replace bitsandbytes Linear4bit layers with standard nn.Linear.

        When loaded on CPU, BNB keeps 4-bit weights either as packed uint8
        (quant_state set) or as full-precision bfloat16 (skip modules). In
        both cases we convert to nn.Linear so the model can run on TT silicon
        without a CUDA-initialized quantization state.
        """
        try:
            import bitsandbytes as bnb
            from bitsandbytes.functional import dequantize_4bit
        except ImportError:
            return model

        for module_path, module in list(model.named_modules()):
            if not isinstance(module, bnb.nn.Linear4bit):
                continue

            w = module.weight
            if w.shape[1] == 1:
                # Packed 4-bit weight — dequantize to bfloat16.
                weight_fp = dequantize_4bit(w.data, w.quant_state).to(torch.bfloat16)
            else:
                # Already full-precision (skip-module layers).
                weight_fp = w.data.to(torch.bfloat16)

            out_f, in_f = weight_fp.shape
            has_bias = module.bias is not None
            new_linear = nn.Linear(in_f, out_f, bias=has_bias, dtype=torch.bfloat16)
            new_linear.weight = nn.Parameter(weight_fp)
            if has_bias:
                new_linear.bias = nn.Parameter(module.bias.data.to(torch.bfloat16))

            if "." in module_path:
                parent_path, child_name = module_path.rsplit(".", 1)
                parent = model.get_submodule(parent_path)
            else:
                parent = model
                child_name = module_path
            setattr(parent, child_name, new_linear)

        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MedGemma multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The MedGemma model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        if self._variant == ModelVariant.UNSLOTH_MEDGEMMA_4B_IT_BNB_4BIT:
            model_kwargs["device_map"] = "cpu"
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if self._variant == ModelVariant.UNSLOTH_MEDGEMMA_4B_IT_BNB_4BIT:
            model = self._dequantize_bnb_linear4bit(model)

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the MedGemma multimodal model.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution.

        Args:
            num_devices: Number of devices to shard across.

        Returns:
            tuple: (mesh_shape, mesh_axis_names) where mesh_shape is (batch_dim, model_dim)
                   and mesh_axis_names are ("batch", "model").
        """
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")
