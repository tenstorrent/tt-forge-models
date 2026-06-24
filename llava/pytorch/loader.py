# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA model loader implementation for multimodal conditional generation.
"""

import os
import re
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available LLaVA model variants."""

    LLAVA_1_5_7B = "1.5_7B"
    LLAVA_1_6_34B = "1.6_34B"


class ModelLoader(ForgeModel):
    """LLaVA model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_1_5_7B: ModelConfig(
            pretrained_model_name="llava-hf/llava-1.5-7b-hf",
        ),
        ModelVariant.LLAVA_1_6_34B: ModelConfig(
            pretrained_model_name="llava-hf/llava-v1.6-34b-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_1_5_7B

    sample_image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    sample_text = "What’s shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA model loader."""
        super().__init__(variant)
        self.processor = None
        self.config = None

    @property
    def _is_next(self) -> bool:
        """LLaVA-NeXT (1.6) uses LlavaNextForConditionalGeneration."""
        return self._variant == ModelVariant.LLAVA_1_6_34B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        group = (
            ModelGroup.GENERALITY
            if variant == ModelVariant.LLAVA_1_6_34B
            else ModelGroup.RED
        )
        return ModelInfo(
            model="LLaVA",
            variant=variant,
            group=group,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA model instance."""
        model_name = self._variant_config.pretrained_model_name
        model_cls = (
            LlavaNextForConditionalGeneration
            if self._is_next
            else LlavaForConditionalGeneration
        )
        model = model_cls.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        self.config = model.config

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, padding=True, add_generation_prompt=True
        )

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        if "image_sizes" in inputs:
            result["image_sizes"] = inputs["image_sizes"]
        return result

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel.

        Decisions are driven by the language model's attention head count.
        """
        n_heads = self.config.text_config.num_attention_heads
        if n_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif n_heads % (num_devices // 2) == 0 and num_devices % 2 == 0:
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {n_heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard spec for the Llama backbone (Yi-34B, GQA).

        Only the language model is sharded; the CLIP vision tower and the
        multimodal projector stay replicated (they're small relative to the
        34B LM). Standard llama column/row split.
        """
        if not self._is_next:
            return None

        lm = getattr(getattr(model, "model", None), "language_model", None) or getattr(
            model, "language_model", None
        )
        layers = lm.layers if hasattr(lm, "layers") else lm.model.layers

        shard_specs = {}
        for layer in layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None:
            shard_specs[lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
