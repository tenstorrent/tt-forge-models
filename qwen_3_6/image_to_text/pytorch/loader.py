# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 (Qwen3.6-27B) model loader implementation for image to text.

Qwen3.6-27B is a multimodal vision-language model
(``Qwen3_5ForConditionalGeneration``, model_type ``qwen3_5``) pairing a
ViT-style vision encoder with a hybrid text backbone that interleaves Gated
DeltaNet linear-attention layers with standard full-attention layers
(``full_attention`` every 4th layer, 16 full / 48 linear out of 64). The
text model uses GQA (24 q : 4 kv heads, head_dim 256), partial RoPE with an
interleaved multimodal RoPE (mRoPE), and an MTP head.
"""
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
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
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Qwen 3.6 model variants for image to text."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_27B

    # Shared configuration parameters
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    sample_text = "Describe this image."
    # Side length the sample image is resized to before processing. The full
    # demo image expands to ~2.7k vision tokens; a small square keeps the
    # sequence length tractable for on-device compilation (~80 tokens / 256
    # patches) while still exercising the full vision + text path. PCC is
    # computed CPU-vs-device on this same input, so a small image is sufficient
    # for validation.
    image_size = 224

    def _get_sample_image(self):
        """Return a PIL image for the sample input, resized to ``image_size``.

        Fetches the demo image; falls back to a deterministic synthetic image
        if the network is unavailable so the loader stays usable offline.
        """
        from PIL import Image

        size = (self.image_size, self.image_size)
        try:
            import requests

            img = Image.open(
                requests.get(self.sample_image, stream=True, timeout=30).raw
            ).convert("RGB")
            return img.resize(size)
        except Exception:
            import numpy as np

            # Deterministic gradient pattern (no RNG) so inputs are reproducible.
            xs = np.linspace(0, 255, self.image_size, dtype="uint8")
            grid = np.tile(xs, (self.image_size, 1))
            arr = np.stack([grid, grid.T, (grid + grid.T) // 2], axis=-1)
            return Image.fromarray(arr, mode="RGB")

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

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
            model="qwen_3_6",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load the processor for this instance's variant."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. If not provided the model is loaded in its
                            native checkpoint dtype (bfloat16).

        Returns:
            torch.nn.Module: The Qwen 3.6 model instance for image to text.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        # Load the model in its native dtype by default ("auto" -> bfloat16),
        # or the requested override.
        model_kwargs = {}
        model_kwargs["dtype"] = dtype_override if dtype_override is not None else "auto"
        model_kwargs |= kwargs

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Force use_cache=False on the live model config so the forward output
        # does not include a Qwen3_5DynamicCache, which the runner's pytree
        # comparator can't diff leaf-wise against the CPU golden. Same pattern
        # as the qwen_3_5 / qwen_2_5_vl loaders — passing use_cache via
        # from_pretrained kwargs is overwritten when the model rebuilds its
        # config from the checkpoint.
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        # Wrap so the model is trace-friendly (pins the vision-tower dtype to
        # avoid the generator-based `.dtype` property that breaks TorchDynamo)
        # and returns only the logits tensor for the runner's comparator.
        model = Wrapper(model)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.6 model.

        Args:
            dtype_override: Optional torch.dtype to override the floating-point
                            inputs' dtype (e.g. pixel_values).
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._get_sample_image()},
                    {"type": "text", "text": self.sample_text},
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

        # Cast floating-point inputs (e.g. pixel_values) to the requested dtype
        # so they match the model weights' dtype on device. Integer inputs
        # (input_ids, attention_mask, image_grid_thw) are left untouched.
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and torch.is_floating_point(
                    inputs[key]
                ):
                    inputs[key] = inputs[key].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
