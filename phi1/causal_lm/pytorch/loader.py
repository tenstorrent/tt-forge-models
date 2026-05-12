# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PHI1 model loader implementation for causal language modeling.
"""
import torch
from transformers import PhiForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel, ForgePrefillModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import (
    get_static_cache_decode_inputs,
)


class ModelVariant(StrEnum):
    """Available PHI1 model variants."""

    PHI1 = "Phi_1"


class ModelLoader(ForgeModel):
    """PHI1 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.PHI1: LLMModelConfig(
            pretrained_model_name="microsoft/phi-1",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PHI1

    # Shared configuration parameters
    sample_text = "Africa is an emerging economy because"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers
        self.model = None
        self.config = None
        self.seq_len = None

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
            model="Phi-1",
            variant=variant,
            group=ModelGroup.RED,
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
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PHI1 model instance for this instance's variant.

        Args:
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The PHI1 model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = PhiForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)

        self.model = model
        self.config = model.config

        return model

    def load_config(self):
        """Load and return the configuration for the Phi-1 model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def load_shard_spec(self, model, strategy="fsdp", batch_axis="batch"):
        """No default TP sharding for Phi-1; see :class:`ModelLoaderPrefill`
        for the parameterized FSDP/Megatron version.
        """
        return {}

    def load_inputs_decode(self, dtype_override=None, batch_size=1):
        """Load decode-step inputs (single token + static KV cache)."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        if self.config is None:
            self.load_config()

        max_cache_len = self._variant_config.max_length
        return get_static_cache_decode_inputs(
            tokenizer=self.tokenizer,
            config=self.config,
            model=self.model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype_override,
        )

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the PHI1 model with this instance's variant settings.
        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Create tokenized inputs for the causal language modeling task
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded prediction for the next tokens
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        # Get the logits from the outputs
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Get the predicted token IDs
        predicted_token_ids = logits.argmax(dim=-1)

        # Decode the predicted tokens
        predicted_text = self.tokenizer.decode(
            predicted_token_ids[0], skip_special_tokens=True
        )

        return predicted_text


class ModelLoaderPrefill(ModelLoader, ForgePrefillModel):
    """Prefill-focused loader for PHI1 variants on which we test prefill
    extensively with various meshes, strategies, batches and sequence lengths.
    """

    _VARIANTS = {
        ModelVariant.PHI1: ModelLoader._VARIANTS[ModelVariant.PHI1],
    }
    DEFAULT_VARIANT = ModelVariant.PHI1

    def load_shard_spec(self, model, strategy="fsdp", batch_axis="batch"):
        """Weight shard spec parameterized by ``strategy`` and ``batch_axis``
        (use "data" when inputs are also sharded).

        Phi-1 differs from Llama-family models: attention output is named
        ``self_attn.dense``, the MLP is a non-gated two-layer ``fc1``/``fc2``,
        the final layer norm lives at ``model.final_layernorm``, and most
        weights have biases.
        """
        shard_specs = {}

        if strategy == "fsdp":
            shard_specs[model.model.embed_tokens.weight] = (None, batch_axis)
            shard_specs[model.lm_head.weight] = ("model", batch_axis)
            shard_specs[model.lm_head.bias] = ("model",)
            shard_specs[model.model.final_layernorm.weight] = (batch_axis,)
            shard_specs[model.model.final_layernorm.bias] = (batch_axis,)
            for layer in model.model.layers:
                shard_specs[layer.mlp.fc1.weight] = ("model", batch_axis)
                shard_specs[layer.mlp.fc1.bias] = ("model",)
                shard_specs[layer.mlp.fc2.weight] = (batch_axis, "model")
                shard_specs[layer.mlp.fc2.bias] = (batch_axis,)

                shard_specs[layer.self_attn.q_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.q_proj.bias] = ("model",)
                shard_specs[layer.self_attn.k_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.k_proj.bias] = ("model",)
                shard_specs[layer.self_attn.v_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.v_proj.bias] = ("model",)
                shard_specs[layer.self_attn.dense.weight] = (batch_axis, "model")
                shard_specs[layer.self_attn.dense.bias] = (batch_axis,)
                shard_specs[layer.input_layernorm.weight] = (batch_axis,)
                shard_specs[layer.input_layernorm.bias] = (batch_axis,)

        elif strategy == "megatron":
            shard_specs[model.model.embed_tokens.weight] = (None, None)
            shard_specs[model.lm_head.weight] = ("model", None)
            shard_specs[model.lm_head.bias] = ("model",)
            shard_specs[model.model.final_layernorm.weight] = (None,)
            shard_specs[model.model.final_layernorm.bias] = (None,)
            for layer in model.model.layers:
                shard_specs[layer.mlp.fc1.weight] = ("model", None)
                shard_specs[layer.mlp.fc1.bias] = ("model",)
                shard_specs[layer.mlp.fc2.weight] = (None, "model")
                shard_specs[layer.mlp.fc2.bias] = (None,)

                shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.q_proj.bias] = ("model",)
                shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.k_proj.bias] = ("model",)
                shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.v_proj.bias] = ("model",)
                shard_specs[layer.self_attn.dense.weight] = (None, "model")
                shard_specs[layer.self_attn.dense.bias] = (None,)
                shard_specs[layer.input_layernorm.weight] = (None,)
                shard_specs[layer.input_layernorm.bias] = (None,)

        else:
            raise ValueError(f"Unknown sharding strategy: {strategy!r}")

        return shard_specs
