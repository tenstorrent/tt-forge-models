# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Olmo3 Causal LM model loader implementation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional
from transformers.cache_utils import StaticCache


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
    """Available Olmo3 model variants for causal language modeling."""

    Olmo_3_7B_Think = "3_7b_think"
    Olmo_3_7B_Instruct = "3_7b_instruct"
    Olmo_3_1025_7B = "3_1025_7b"
    Olmo_3_32B_Think = "3_32b_think"
    Olmo_3_1125_32B = "3_1125_32b"


class ModelLoader(ForgeModel):
    """Olmo3 model loader implementation for causal language modeling tasks."""

    # These variants use sliding window attention and need StaticCache + overrides.
    _SLIDING_WINDOW_VARIANTS = {
        ModelVariant.Olmo_3_1025_7B,
        ModelVariant.Olmo_3_1125_32B,
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Olmo_3_7B_Think: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-7B-Think",
            max_length=256,
        ),
        ModelVariant.Olmo_3_7B_Instruct: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-7B-Instruct",
            max_length=256,
        ),
        ModelVariant.Olmo_3_1025_7B: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-1025-7B",
            max_length=256,
        ),
        ModelVariant.Olmo_3_1125_32B: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-1125-32B",
            max_length=256,
        ),
        ModelVariant.Olmo_3_32B_Think: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-32B-Think",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Olmo_3_7B_Think

    # Shared configuration parameters
    sample_text = "Who would win in a fight - a dinosaur or a cow named Moo Moo?"

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

        group = ModelGroup.RED
        return ModelInfo(
            model="olmo_3",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Olmo 3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Olmo 3 model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if self._variant in self._SLIDING_WINDOW_VARIANTS:
            from tt_torch.transformers_overrides import (
                override_olmo3_sliding_window_causal_mask,
            )

            override_olmo3_sliding_window_causal_mask()
        model.eval()
        print("model", model)
        print("model.config", model.config)
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Olmo 3 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_cache_len = self._variant_config.max_length

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

        # Compute the actual prompt token length to get a consistent static shape
        prompt_token_len = len(
            self.tokenizer.encode(self.sample_text, add_special_tokens=False)
        )

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

        # Non-sliding variants: return standard tokenizer output
        if self._variant not in self._SLIDING_WINDOW_VARIANTS:
            print("inputs", inputs)
            return inputs

        # Sliding window variants: build full input_args with StaticCache
        seq_len = inputs.input_ids.shape[1]

        from tt_torch.transformers_overrides import (
            override_cache_sliding_window_layers,
            _init_static_cache,
        )

        static_cache = StaticCache(
            config=self.config,
            max_cache_len=max_cache_len,
        )
        _init_static_cache(static_cache, self.config, batch_size)

        sliding_window = getattr(
            self.config.get_text_config(decoder=True), "sliding_window", max_cache_len
        )
        override_cache_sliding_window_layers(
            static_cache, max_cache_len, sliding_window
        )

        # Attention mask must match max_cache_len to prevent recompilation or
        # implicit padding by transformers, which can cause degenerate output.
        full_attention_mask = torch.ones(
            (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
        )
        full_attention_mask[:, :seq_len] = inputs.attention_mask

        cache_position = torch.arange(0, seq_len)

        input_args = {
            "input_ids": inputs.input_ids,
            "past_key_values": static_cache,
            "cache_position": cache_position,
            "use_cache": True,
            "attention_mask": full_attention_mask,
        }
        print("input_args", input_args)
        return input_args

    def get_mesh_config(self, num_devices: int):

        # Prefer (1, N) when heads divide N, otherwise try (2, N/2)
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
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
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("batch", "model")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Olmo 3 model variant.

        Returns:
            The configuration object for the Olmo 3 model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
