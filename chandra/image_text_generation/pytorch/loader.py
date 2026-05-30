# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chandra OCR model loader implementation for image-to-text (document OCR) tasks.

Source model: prithivMLmods/chandra-OCR-GGUF — a llama.cpp GGUF quantization of
datalab-to/chandra, a Qwen3-VL-8B document-OCR / VLM fine-tune (architecture
``qwen3vl`` for the language model + a ``clip`` ``qwen3vl_merger`` mmproj vision
tower).

transformers' GGUF dequantization path does not support the ``qwen3vl``
architecture (it is absent from ``GGUF_CONFIG_MAPPING`` /
``GGUF_SUPPORTED_ARCHITECTURES``) and cannot load the separate ``clip`` mmproj
vision tower, so the GGUF blobs cannot be turned back into a PyTorch model via
``from_pretrained(gguf_file=...)``. The loader therefore loads the canonical
upstream weights from ``datalab-to/chandra`` (the exact, un-quantized model the
GGUF was produced from — identical architecture and, for the BF16/F16 GGUF
files, identical weights).
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Chandra OCR model variants."""

    OCR = "ocr"


class ModelLoader(ForgeModel):
    """Chandra OCR model loader implementation for image-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OCR: LLMModelConfig(
            # Canonical (un-quantized) weights for prithivMLmods/chandra-OCR-GGUF.
            pretrained_model_name="datalab-to/chandra",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OCR

    # Shared configuration parameters
    sample_image = (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/pipeline-cat-chonk.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="chandra",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load the processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance
        """
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Chandra OCR model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its default dtype.

        Returns:
            torch.nn.Module: The Chandra OCR model instance for image-to-text tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.config = model.config

        # TorchDynamo workaround: Qwen3-VL's vision tower computes its working
        # dtype via the inherited `PreTrainedModel.dtype` property, which does
        # `next(p.dtype for p in self.parameters() if p.is_floating_point())`.
        # Tracing that generator over module parameters trips a known dynamo bug
        # ("cannot access free variable 'named_children'") inside
        # `get_image_features` (`pixel_values.type(self.visual.dtype)`). The
        # weights have a single fixed dtype, so replace the property with a
        # constant the compiler can fold.
        visual = getattr(getattr(model, "model", model), "visual", None)
        if visual is not None:
            try:
                visual_dtype = next(
                    p.dtype for p in visual.parameters() if p.is_floating_point()
                )
                type(visual).dtype = property(lambda self, _d=visual_dtype: _d)
            except StopIteration:
                pass

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Chandra OCR model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image},
                    {"type": "text", "text": "Extract the text from this image."},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def load_config(self):
        """Load and return the configuration for the Chandra OCR model variant.

        Returns:
            The configuration object for the Chandra OCR model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
