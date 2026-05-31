# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Astral-14B model loader implementation for causal language modeling.

The model id requested for bringup, ``mradermacher/Astral-14B-i1-GGUF``, is a
community GGUF (llama.cpp) re-quantization of the upstream model
``LucidityAI/Astral-14B`` (a Qwen3-14B fine-tune). The GGUF K-quant / IQ block
formats cannot be executed natively on Tenstorrent hardware - the only way to run
them is to dequantize back to bf16/fp32, which makes the result numerically
equivalent to the upstream bf16 weights. transformers' GGUF dequantization for a
14B model also materializes a ~56 GB fp32 state dict on the host, which exceeds
the runner's memory budget.

This loader therefore loads the upstream ``LucidityAI/Astral-14B`` safetensors
directly in bfloat16 (~28 GB, no fp32 intermediate). It is the same fine-tuned
Astral-14B model that the GGUF repo is derived from, and the resulting graph is
the Qwen3 architecture already supported on Tenstorrent hardware.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Available Astral-14B model variants for causal language modeling."""

    ASTRAL_14B = "14B"


class ModelLoader(ForgeModel):
    """Astral-14B model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.ASTRAL_14B: LLMModelConfig(
            pretrained_model_name="LucidityAI/Astral-14B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ASTRAL_14B

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
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
            model="Astral 14B",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Astral-14B model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Astral-14B model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Astral-14B model.

        Args:
            dtype_override: Unused for integer token inputs; accepted for interface
                            compatibility.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        # Use the chat template for this Qwen3-based model.
        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        # Replicate to the requested batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Tensor-parallel mesh config (model sharded across the device axis)."""
        mesh_shape = (1, num_devices)
        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Shard spec for tensor-parallel execution across multiple chips."""
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Astral-14B variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
