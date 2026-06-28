# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 model loader implementation for image-text-to-text (vision-language).

Qwen/Qwen3.6-27B is a ``Qwen3_5ForConditionalGeneration`` vision-language model:
a SigLIP-style vision tower (depth 27, hidden 1152, patch 16, spatial-merge 2)
feeds a Qwen3.5 hybrid text decoder. The decoder interleaves Gated DeltaNet
linear-attention layers (causal conv1d + chunked delta rule) with standard full
attention in a (3x linear + 1x full) pattern over 64 layers, and uses interleaved
multimodal RoPE (mrope) with a 0.25 partial-rotary factor.
"""
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
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
    """Available Qwen 3.6 model variants for image-text-to-text."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_27B

    sample_prompt = "Describe this image."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional override for the text decoder depth. Used for
                        op-coverage pre-checks and to fit the model on device
                        without the full 64-layer footprint.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3.6",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def _sample_image(self):
        """Deterministic offline RGB test image (a smooth gradient).

        Avoids a network dependency in CI; the processor only needs a valid
        ``PIL.Image`` to exercise the patch-embed / vision-tower path.
        """
        import numpy as np

        h = w = 256
        x = np.linspace(0, 255, w, dtype=np.uint8)
        y = np.linspace(0, 255, h, dtype=np.uint8)
        r = np.tile(x, (h, 1))
        g = np.tile(y[:, None], (1, w))
        b = ((r.astype(int) + g.astype(int)) // 2).astype(np.uint8)
        arr = np.stack([r, g, b], axis=-1)
        return Image.fromarray(arr, mode="RGB")

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 VLM instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype (the checkpoint ships in bfloat16).

        Returns:
            torch.nn.Module: The Qwen 3.6 image-text-to-text model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.text_config.num_hidden_layers = self.num_layers
            # Keep at least one full-attention layer so both branches of the
            # hybrid stack are exercised even when reduced.
            config.text_config.layer_types = config.text_config.layer_types[
                : self.num_layers
            ]
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output does not include a
        # DynamicCache, which the runner's pytree comparator can't diff
        # leaf-wise against the CPU golden (same pattern as the qwen_3_5
        # and qwen_2_5_vl loaders).
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.6 VLM.

        Args:
            dtype_override: Optional torch.dtype; floating-point inputs
                            (e.g. pixel_values) are cast to match.
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
                    {"type": "image", "image": self._sample_image()},
                    {"type": "text", "text": self.sample_prompt},
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

        return dict(inputs)

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
