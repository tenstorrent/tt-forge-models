# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.6 (Qwen3.6-35B-A3B) model loader for image-text-to-text.

Qwen3.6 is a multimodal Mixture-of-Experts VLM:
  * Text backbone: ``qwen3_5_moe`` — a hybrid decoder that interleaves
    Gated DeltaNet linear attention (causal conv1d + chunked delta rule)
    with standard full attention every ``full_attention_interval`` (=4)
    layers, on top of a 256-expert / 8-active sparse MoE FFN.
  * Vision tower: a Qwen3-VL style SigLIP-derived encoder (patch conv,
    spatial 2x2 patch merging) producing visual tokens fused into the
    text stream.

A single ``ModelLoader`` loads the full ``Qwen3_5MoeForConditionalGeneration``
pipeline plus its processor, mirroring the qwen_3_vl loader. ``num_layers`` /
``vision_depth`` overrides allow a reduced-depth forward for fast CPU/op
bring-up probes without changing the default (full-size) variant.
"""
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3_5MoeForConditionalGeneration,
)

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen 3.6 model variants for image-text-to-text."""

    QWEN_3_6_35B_A3B = "35b_a3b"


class ModelLoader(ForgeModel):
    """Qwen 3.6 model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-35B-A3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_35B_A3B

    # Fixed prompt used for both the loader inputs and the PoC artifact.
    sample_prompt = "Describe this image."

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        num_layers: Optional[int] = None,
        vision_depth: Optional[int] = None,
    ):
        """Initialize ModelLoader.

        Args:
            variant: Which ModelVariant to use (defaults to DEFAULT_VARIANT).
            num_layers: Optional override for the text backbone's
                ``num_hidden_layers`` — used for reduced-depth bring-up probes.
            vision_depth: Optional override for the vision tower's ``depth``.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers
        self.vision_depth = vision_depth

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
        """Load and return the Qwen 3.6 model instance for this variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Optional reduced-depth config for fast CPU / op-support probes.
        if self.num_layers is not None or self.vision_depth is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if self.num_layers is not None:
                config.text_config.num_hidden_layers = self.num_layers
                # Keep the interleaved linear/full-attention layer_types list
                # consistent with the reduced layer count.
                if getattr(config.text_config, "layer_types", None) is not None:
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            if self.vision_depth is not None:
                config.vision_config.depth = self.vision_depth
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Force use_cache=False so the forward output does not include a
        # dynamic cache object that the runner's pytree comparator cannot
        # diff leaf-wise against the CPU golden (same pattern as qwen_3_5 /
        # qwen_2_5_vl loaders).
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return sample image-text inputs for the Qwen 3.6 model."""
        if self.processor is None:
            self._load_processor()

        # Use a locally-generated image so the bring-up does not depend on
        # fetching a remote URL at test time.
        image = Image.new("RGB", (448, 448), color=(73, 109, 137))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        for key in inputs:
            if torch.is_tensor(inputs[key]) and inputs[key].dim() > 0 and batch_size > 1:
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
