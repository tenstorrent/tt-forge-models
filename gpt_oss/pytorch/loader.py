# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
gpt-oss model loader implementation for causal language modeling tasks.
"""
import types
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast
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
    """Available gpt-oss model variants."""

    GPT_OSS_20B = "gpt_oss_20b"
    GPT_OSS_120B = "gpt_oss_120b"

NUM_LAYERS_TO_RUN = 16  # Number of layers to execute during forward pass


def limited_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    cache_position=None,
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Only iterate through the first NUM_LAYERS_TO_RUN layers
    for decoder_layer in self.layers[:NUM_LAYERS_TO_RUN]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )

class ModelLoader(ForgeModel):
    """gpt-oss model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.GPT_OSS_20B: LLMModelConfig(
            pretrained_model_name="openai/gpt-oss-20b",
            max_length=256,
        ),
        ModelVariant.GPT_OSS_120B: LLMModelConfig(
            pretrained_model_name="openai/gpt-oss-120b",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B

    # Sample messages for inference
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

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
        self.config = None
        self.tokenizer = None
        self.num_layers = num_layers

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
            model="gpt_oss",
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
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the gpt-oss model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16 as default.

        Returns:
            torch.nn.Module: The gpt-oss model instance for causal language modeling.
        """
        # Load config with modifications
        self.load_config()

        # Prepare model kwargs
        model_kwargs = {
            "config": self.config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        # Set dtype - default to bfloat16 if not specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Replace forward to only execute limited layers (uses NUM_LAYERS_TO_RUN constant)
        model.model.forward = types.MethodType(limited_forward, model.model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the gpt-oss model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           This is currently not used as tokenized inputs are integers.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Create tokenized inputs
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
        )

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get mesh configuration for tensor parallelism.

        Args:
            num_devices: Number of devices to use for tensor parallelism

        Returns:
            Tuple of (mesh_shape, axis_names)
        """
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load shard specifications for tensor parallelism.

        Args:
            model: The gpt-oss model instance

        Returns:
            Dictionary mapping model parameters to their shard specifications,
            or None if sharding is not needed for this variant
        """
        shard_specs = {}
        for layer in model.model.layers:
            # Self-attention weights
            # q_proj, k_proj, v_proj: column-wise sharding
            shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
            # o_proj: row-wise sharding
            shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
            shard_specs[layer.self_attn.sinks] = (None,)

            # MoE MLP components
            # Router is replicated across all devices
            shard_specs[layer.mlp.router.weight] = (None, None)

            # Expert weights - sharded across the expert dimension
            # These are 3D tensors with shape (num_experts, hidden_size, intermediate_size)
            shard_specs[layer.mlp.experts.gate_up_proj] = (
                "model",
                None,
                None,
            )
            shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
            shard_specs[layer.mlp.experts.down_proj] = (
                "model",
                None,
                None,
            )
            shard_specs[layer.mlp.experts.down_proj_bias] = ("model", None)

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the gpt-oss model with this instance's variant.

        Returns:
            The configuration object for the gpt-oss model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers

        self.config.quantization_config["quant_method"] = "none"
        self.config.use_cache = False
        return self.config
