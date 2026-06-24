# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bloom model loader implementation
"""


import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available BLOOM model variants."""

    BLOOM_1B1 = "1b1"
    BLOOM_176B = "176B"


class ModelLoader(ForgeModel):
    """Bloom model loader implementation."""

    _VARIANTS = {
        ModelVariant.BLOOM_1B1: ModelConfig(
            pretrained_model_name="bigscience/bloom-1b1",
        ),
        ModelVariant.BLOOM_176B: ModelConfig(
            pretrained_model_name="bigscience/bloom",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BLOOM_1B1

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)

        self.tokenizer = None
        self.config = None
        self.model = None
        self.test_input = "This is a sample text from "
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BLOOM",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Bloom model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           For the 176B variant, defaults to bfloat16.

        Returns:
            torch.nn.Module: The Bloom model instance.
        """
        model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        elif self._variant == ModelVariant.BLOOM_176B:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        self.config = model.config
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Bloom model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """

        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        if self._variant == ModelVariant.BLOOM_176B:
            inputs = self.tokenizer(
                self.test_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
        else:
            inputs = self.tokenizer(
                self.test_input,
                return_tensors="pt",
                max_length=32,
                padding="max_length",
                add_special_tokens=True,
                truncation=True,
            )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded answer text
        """
        if self.tokenizer is None:
            self.load_model()
        next_token_logits = outputs.logits[:, -1]
        next_tokens = next_token_logits.softmax(dim=-1).argmax(dim=-1)
        return [self.tokenizer.decode([token.item()]) for token in next_tokens]

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        elif self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads "
                f"across {num_devices} devices"
            )
        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Return tensor shard specs for the BLOOM architecture."""
        if self._variant != ModelVariant.BLOOM_176B:
            return None

        shard_specs = {}
        for layer in model.transformer.h:
            shard_specs[layer.mlp.dense_h_to_4h.weight] = ("model", "batch")
            shard_specs[layer.mlp.dense_h_to_4h.bias] = ("model",)
            shard_specs[layer.mlp.dense_4h_to_h.weight] = ("batch", "model")

            shard_specs[layer.self_attention.query_key_value.weight] = (
                "model",
                "batch",
            )
            shard_specs[layer.self_attention.query_key_value.bias] = ("model",)
            shard_specs[layer.self_attention.dense.weight] = ("batch", "model")

        shard_specs[model.transformer.word_embeddings.weight] = ("model", "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
