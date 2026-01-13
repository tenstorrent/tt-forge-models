# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Pixtral model loader implementation
"""
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from typing import Optional
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Pixtral model variants."""

    PIXTRAL_12B = "pixtral-12b"


class ModelLoader(ForgeModel):
    """Pixtral model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.PIXTRAL_12B: ModelConfig(
            pretrained_model_name="mistral-community/pixtral-12b",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PIXTRAL_12B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="mistral-community",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Mistral Pixtral model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Mistral Pixtral model instance.

        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, batch_size=1, dtype_override=None):
        """Load and return sample inputs for the Mistral Pixtral model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)
        url_dog = "https://picsum.photos/id/237/200/300"
        url_mountain = "https://picsum.photos/seed/picsum/200/300"
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": "Can this animal"},
                    {"type": "image"},
                    {"type": "text", "content": "live here?"},
                    {"type": "image"},
                ],
            }
        ]
        prompt = self.tokenizer.apply_chat_template(chat)
        inputs = self.tokenizer(
            text=prompt, images=[url_dog, url_mountain], return_tensors="pt"
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)
        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        for layer in model.vision_tower.transformer.layers:
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")

            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

        return shard_specs
