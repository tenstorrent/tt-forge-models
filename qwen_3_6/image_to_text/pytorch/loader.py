# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 (Qwen3.6-35B-A3B) model loader implementation for image-to-text.

Qwen3.6-35B-A3B is a multimodal (vision + text) Mixture-of-Experts model with
the HuggingFace ``qwen3_5_moe`` architecture
(``Qwen3_5MoeForConditionalGeneration``). It pairs:

* a SigLIP-style vision tower (depth 27, hidden 1152, patch 16, spatial-merge 2,
  temporal-patch 2, gelu_pytorch_tanh) that projects to the 2048-d text space, and
* a hybrid linear/full-attention MoE text decoder (40 layers, 3x Gated-DeltaNet
  linear-attention + 1x full-attention interleaved, 256 experts / 8 active per
  token, shared expert, mrope) with a 248320 vocab.

35B total params, ~3B active per token (A3B). Weights are distributed in bf16.
"""
import torch
from transformers import (
    Qwen3_5MoeForConditionalGeneration,
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
    """Available Qwen 3.6 model variants for image-to-text."""

    QWEN_3_6_35B_A3B = "35b_a3b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 VLM model loader implementation for image-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_6_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-35B-A3B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_35B_A3B

    # Shared configuration parameters
    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    sample_text = "Describe this image."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional override for the text decoder layer count. The
                     full 40-layer / 256-expert model does not fit a single
                     32 GB chip; reducing the layer count (must keep at least
                     one ``full_attention`` layer, i.e. >= 4) yields a graph
                     that exercises every op type while fitting on device.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.6 VLM model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. Defaults to the checkpoint dtype (bf16).

        Returns:
            torch.nn.Module: The Qwen 3.6 model instance for image-to-text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Optionally shrink the text decoder so a layer-reduced graph fits a
        # single chip (the full 35B model needs multi-chip sharding).
        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.text_config.num_hidden_layers = self.num_layers
            # Keep the interleave pattern consistent with the reduced depth.
            lt = config.text_config.layer_types[: self.num_layers]
            config.text_config.layer_types = lt
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output does not include a
        # DynamicCache, which the runner's pytree comparator cannot diff
        # leaf-wise against the CPU golden (same pattern as qwen_2_5_vl).
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.6 VLM.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors (input_ids, attention_mask, pixel_values, ...).
        """
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image},
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

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
