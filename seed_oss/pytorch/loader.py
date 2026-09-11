# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Seed-OSS model loader implementation.
"""

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Seed-OSS model variants for causal language modeling."""

    SEED_OSS_36B_INSTRUCT = "Seed-OSS-36B-Instruct"


class ModelLoader(ForgeModel):
    """Seed-OSS model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SEED_OSS_36B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="ByteDance-Seed/Seed-OSS-36B-Instruct",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEED_OSS_36B_INSTRUCT

    sample_text = "How to make pasta?"

    # Seed-OSS exposes a thinking budget through its chat template, which
    # prepends a system prompt instructing the model to bound its reasoning.
    thinking_budget = 512

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Seed-OSS-36B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.tokenizer

    def load_config(self):
        """Load and return the configuration for this instance's variant.

        Returns:
            The configuration object for the Seed-OSS model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Seed-OSS model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Seed-OSS model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Seed-OSS model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        conversation = [{"role": "user", "content": self.sample_text}]
        template_kwargs = {}
        if self.thinking_budget is not None:
            template_kwargs["thinking_budget"] = self.thinking_budget
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        if self.config is None:
            self.load_config()

        # Under GQA the key/value projections are the narrow ones (8 kv heads
        # here vs 80 query heads), so they set the ceiling on the model axis.
        assert (
            self.config.num_key_value_heads % mesh_shape[1] == 0
        ), "Key/value heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            # This checkpoint sets attention_bias=true, so q/k/v each carry a
            # bias that has to follow its weight's output-dim shard.
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            mlp = layer.mlp
            shard_specs[mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[mlp.up_proj.weight] = ("model", "batch")
            shard_specs[mlp.down_proj.weight] = ("batch", "model")

        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
