# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hemlock Qwen 2.5 Coder model loader implementation

Hemlock-Qwen2.5-Coder-7B is an SFT fine-tune of Qwen/Qwen2.5-Coder-7B
(nbeerbower/Hemlock-Qwen2.5-Coder-7B). The published weights here are
GGUF (imatrix "i1") quantizations from mradermacher. transformers loads a
GGUF checkpoint by dequantizing it back into a standard Qwen2 PyTorch module
via the ``gguf_file`` argument, so the rest of the bringup flow is identical
to a regular Qwen 2.5 Coder loader.
"""


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    """Available Hemlock Qwen 2.5 Coder model variants for causal language modeling."""

    HEMLOCK_7B = "7b"


class ModelLoader(ForgeModel):
    """Hemlock Qwen 2.5 Coder model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.HEMLOCK_7B: LLMModelConfig(
            pretrained_model_name="mradermacher/Hemlock-Qwen2.5-Coder-7B-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HEMLOCK_7B

    # GGUF checkpoint to dequantize per variant. The i1-Q4_K_M imatrix quant is a
    # good size/accuracy trade-off and uses only Q4_K/Q6_K tensor types, both of
    # which transformers can dequantize.
    _GGUF_FILES = {
        ModelVariant.HEMLOCK_7B: "Hemlock-Qwen2.5-Coder-7B.i1-Q4_K_M.gguf",
    }

    # Shared configuration parameters
    sample_text = "write a quick sort algorithm."

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
            model="Hemlock Qwen 2.5 Coder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self):
        """Return the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hemlock Qwen 2.5 Coder model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The Hemlock Qwen 2.5 Coder model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Store config for mesh/sharding validation
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Hemlock Qwen 2.5 Coder model with this instance's variant settings.

        Args:
            dtype_override: Unused for token inputs (kept for interface compatibility).
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # The published Hemlock checkpoint is a base/SFT coder model; the
        # GGUF-converted tokenizer may not carry a chat template, so tokenize the
        # raw prompt directly for a robust, deterministic input.
        prompts = [self.sample_text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
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
