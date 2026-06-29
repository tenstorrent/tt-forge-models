# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 model loader implementation for image-to-text (vision-language).

Qwen3.6-27B is a vision-language model exposed by transformers as
``Qwen3_5ForConditionalGeneration`` (config ``model_type='qwen3_5'``). It pairs

  * a SigLIP-style ViT vision tower (``qwen3_5`` vision config: depth 27,
    hidden 1152, patch 16, temporal_patch_size 2 -> Conv3d patch embed,
    spatial_merge_size 2, out_hidden_size 5120), and
  * a hybrid text decoder (``qwen3_5_text``: 64 layers interleaving Gated
    DeltaNet linear-attention with causal conv1d and standard full attention
    every 4th layer; hidden 5120, head_dim 256, GQA 24:4, vocab 248320).

A ``num_layers`` override (text-decoder layers) is provided so the loader can be
exercised cheaply on host/device without materializing all 64 layers.
"""
import torch
from PIL import Image
from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
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
    """Available Qwen 3.6 model variants for image to text."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 vision-language model loader for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_27B

    sample_text = "Describe this image."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader.

        Args:
            variant: Optional ModelVariant. If None, DEFAULT_VARIANT is used.
            num_layers: Optional override for the number of text-decoder hidden
                layers (for cheap, reduced-size host/device validation).
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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 VLM instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                dtype. If not provided, the model's checkpoint dtype (bf16) is used.

        Returns:
            torch.nn.Module: The Qwen 3.6 image-to-text model instance.
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
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output does not include a
        # dynamic cache, which the runner's pytree comparator can't diff
        # leaf-wise against the CPU golden (same pattern as qwen_3_5 /
        # qwen_2_5_vl loaders).
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample image+text inputs for the Qwen 3.6 VLM.

        Args:
            dtype_override: Optional torch.dtype to cast floating-point inputs
                (e.g. pixel_values) to match the model dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors (input_ids, attention_mask, pixel_values,
                image_grid_thw) that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        # Locally-generated image avoids a network dependency at test time.
        image = Image.new("RGB", (448, 448), color=(73, 109, 137))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and torch.is_floating_point(
                    inputs[key]
                ):
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
