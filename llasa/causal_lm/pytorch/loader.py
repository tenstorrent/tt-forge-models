# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llasa model loader implementation for causal language modeling.

Llasa (HKUSTAudio/Llasa-8B) is a text-to-speech model built on the
Llama-3.1-8B-Instruct architecture. It is a ``LlamaForCausalLM`` whose
vocabulary is extended with XCodec2 speech tokens; speech is produced by
autoregressively generating speech tokens that an external XCodec2 vocoder
decodes to audio. The compute brought up here is the Llama causal-LM
backbone forward pass, which is what runs on device.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional
import torch

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
)


class ModelVariant(StrEnum):
    """Available Llasa model variants for causal LM."""

    LLASA_8B = "8B"


class ModelLoader(ForgeModel):
    """Llasa model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LLASA_8B: LLMModelConfig(
            pretrained_model_name="HKUSTAudio/Llasa-8B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLASA_8B

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

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
        self.seq_len = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Llasa",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token for Llama-based models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, num_layers=None, **kwargs):
        """Load and return the Llasa model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its config dtype (bfloat16).
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.

        Returns:
            torch.nn.Module: The Llasa model instance for causal LM.
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

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if num_layers is not None:
            model.model.layers = model.model.layers[:num_layers]

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llasa model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # For causal LM, we need both input_ids and attention_mask
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Mesh shape for tensor-parallel sharding across the available devices.

        Llasa-8B (~8.5B params, ~17 GB in bf16) does not fit on a single 12 GB
        Wormhole chip, so it is sharded across all available chips on a
        ("batch", "model") mesh.
        """
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron tensor-parallel weight shard spec on a ("batch", "model") mesh.

        Llasa-8B (~17 GB bf16) does not fit on a single 12 GB Wormhole chip, so
        it is sharded across the n300's two chips with the standard Llama-family
        Megatron layout (column-parallel q/k/v/gate/up + lm_head, row-parallel
        o_proj/down_proj; one all-reduce per attention/MLP pair, intermediates
        never gathered). GQA is consistent at TP=2: the 8 KV heads and 32 Q
        heads both split evenly across the two chips. Embedding and norms are
        replicated. This brings the footprint to ~9.4 GB / 12 GB per chip.
        """
        shard_specs = {}
        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        shard_specs[model.model.norm.weight] = ("batch",)
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Llasa model variant.

        Returns:
            The configuration object for the Llasa model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
