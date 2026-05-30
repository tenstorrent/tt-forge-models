# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OlympicCoder (GGUF) model loader implementation

OlympicCoder-7B is a code-reasoning model fine-tuned from Qwen2.5-Coder-7B-Instruct
(open-r1/OlympicCoder-7B). This loader consumes the GGUF-quantized distribution
published by ``bartowski/open-r1_OlympicCoder-7B-GGUF``. transformers reads the GGUF
container via the ``gguf_file`` argument, dequantizes the weights back into a standard
Qwen2 ``torch.nn.Module`` and reconstructs the tokenizer from the embedded GGUF metadata.
"""


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    """Available OlympicCoder GGUF model variants for causal language modeling."""

    Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """OlympicCoder GGUF model loader implementation for causal language modeling tasks."""

    # HuggingFace repo that hosts the GGUF files.
    _GGUF_REPO = "bartowski/open-r1_OlympicCoder-7B-GGUF"

    # Mapping of variant -> GGUF file name inside the repo. The dequantized
    # architecture is identical across quantization levels; Q4_K_M is the
    # widely recommended general-purpose quant.
    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "open-r1_OlympicCoder-7B-Q4_K_M.gguf",
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/open-r1_OlympicCoder-7B-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    # Shared configuration parameters
    sample_text = "Write a Python function that returns the n-th Fibonacci number."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

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
            model="OlympicCoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """Return the GGUF file name for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF metadata.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._GGUF_REPO,
            gguf_file=self._gguf_file(),
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OlympicCoder model instance for this instance's variant.

        The GGUF checkpoint is dequantized into a standard Qwen2 ``torch.nn.Module``.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.
                            If not provided, the dequantized weights keep their
                            default dtype (float32).

        Returns:
            torch.nn.Module: The OlympicCoder model instance for causal language modeling.
        """
        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(self._GGUF_REPO, **model_kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        # Store config for mesh/sharding validation
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the OlympicCoder model with this instance's variant settings.

        Args:
            dtype_override: Unused for token id inputs; kept for API parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def get_mesh_config(self, num_devices: int):
        # Prefer (1, N) when heads divide N, otherwise try (2, N/2)
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            num_devices % 2 == 0
            and self.config.num_attention_heads % (num_devices // 2) == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
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
