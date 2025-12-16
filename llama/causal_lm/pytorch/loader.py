# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional
import torch
from transformers.cache_utils import StaticCache
import os

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
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Llama model variants for causal LM."""

    # Llama 3 variants
    LLAMA_3_8B = "llama_3_8b"
    LLAMA_3_8B_INSTRUCT = "llama_3_8b_instruct"

    # Llama 3.1 variants
    LLAMA_3_1_8B = "llama_3_1_8b"
    LLAMA_3_1_8B_INSTRUCT = "llama_3_1_8b_instruct"
    LLAMA_3_1_70B = "llama_3_1_70b"
    LLAMA_3_1_70B_INSTRUCT = "llama_3_1_70b_instruct"
    LLAMA_3_1_405B = "llama_3_1_405b"
    LLAMA_3_1_405B_INSTRUCT = "llama_3_1_405b_instruct"

    # Llama 3.2 variants
    LLAMA_3_2_1B = "llama_3_2_1b"
    LLAMA_3_2_1B_INSTRUCT = "llama_3_2_1b_instruct"
    LLAMA_3_2_3B = "llama_3_2_3b"
    LLAMA_3_2_3B_INSTRUCT = "llama_3_2_3b_instruct"

    # Llama 3.3 variants
    LLAMA_3_3_70B_INSTRUCT = "llama_3_3_70b_instruct"

    # HuggingFace community variants
    HUGGYLLAMA_7B = "huggyllama_7b"

    # TinyLlama variants
    TINYLLAMA_V1_1 = "TinyLlama_v1.1"


class ModelLoader(ForgeModel):
    """Llama model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Llama 3 variants
        ModelVariant.LLAMA_3_8B: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3-8B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_length=128,
        ),
        # Llama 3.1 variants
        ModelVariant.LLAMA_3_1_8B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.1-8B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_70B: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-70B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_70B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_405B: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-405B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_405B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-405B-Instruct",
            max_length=128,
        ),
        # Llama 3.2 variants
        ModelVariant.LLAMA_3_2_1B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-1B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_1B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-1B-Instruct",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_3B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-3B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-3B-Instruct",
            max_length=128,
        ),
        # Llama 3.3 variants
        ModelVariant.LLAMA_3_3_70B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.3-70B-Instruct",
            max_length=128,
        ),
        # HuggingFace community variants
        ModelVariant.HUGGYLLAMA_7B: LLMModelConfig(
            pretrained_model_name="huggyllama/llama-7b",
            max_length=128,
        ),
        # TinyLlama variants
        ModelVariant.TINYLLAMA_V1_1: LLMModelConfig(
            pretrained_model_name="TinyLlama/TinyLlama_v1.1",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_1B_INSTRUCT

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Set group based on variant (instruct variants are RED priority except llama_3_8b_instruct and llama_3_1_405b_instruct variant)
        if (
            (
                "instruct" in variant.value
                and (
                    variant
                    not in [
                        ModelVariant.LLAMA_3_8B_INSTRUCT,
                        ModelVariant.LLAMA_3_1_405B_INSTRUCT,
                    ]
                )
            )
            or "70b" in variant.value
            or variant == ModelVariant.LLAMA_3_1_405B
        ):
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="llama_causal_lm",
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
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token for Llama models
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Llama model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
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

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llama model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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

    def load_inputs_decode(self, dtype_override=None, batch_size=1):
        """Load decode-step inputs for Llama (single token + static KV cache).

        This prepares inputs for an autoregressive decode step without running prefill.

        Args:
            dtype_override: Optional torch.dtype for cache tensors (e.g., torch.bfloat16).
            batch_size: Number of sequences (replicates a single decode token).

        Returns:
            dict: Input tensors suitable for a single decode forward pass:
                - input_ids: (batch_size, 1) long
                - attention_mask: (batch_size, 1) long
                - past_key_values: StaticCache
                - cache_position: (1,) long
                - use_cache: bool
        """
        # Ensure tokenizer and config are initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        if getattr(self, "config", None) is None:
            self.load_config()

        # Create zero-initialized static cache for decode
        cache_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        max_cache_len = self._variant_config.max_length
        static_cache = StaticCache(
            config=self.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=cache_dtype,
        )

        # Single decode token per batch (use eos token id)
        eos_id = self.tokenizer.eos_token_id
        input_ids = torch.full((batch_size, 1), fill_value=eos_id, dtype=torch.long)

        # Minimal attention mask for the single token
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long)

        # Choose decode write position:
        # - For pure decode perf testing, write at the end of the cache buffer.
        # - To simulate first-token decode after prefill length k, set to k instead.
        cache_position = torch.tensor([max_cache_len - 1], dtype=torch.long)

        # For compatibility with downstream helpers that may expect seq_len
        self.seq_len = 1

        # Optional debug (off by default). Enable with TT_LLM_DEBUG_DECODE_INPUTS=1
        if os.environ.get("TT_LLM_DEBUG_DECODE_INPUTS") == "1":
            print("\n=== DEBUG: Llama decode input loader ===")
            print(f"Variant: {self._variant}")
            print(f"Batch size: {batch_size} | Max cache len: {max_cache_len}")
            print(f"Cache dtype: {cache_dtype} | Cache device: cpu")
            try:
                num_layers = len(static_cache.key_cache)
            except Exception:
                num_layers = -1
            print(f"StaticCache layers: {num_layers}")
            if (
                isinstance(getattr(static_cache, "key_cache", None), list)
                and len(static_cache.key_cache) > 0
            ):
                try:
                    print(f"key_cache[0].shape: {static_cache.key_cache[0].shape}")
                    print(f"value_cache[0].shape: {static_cache.value_cache[0].shape}")
                except Exception as e:
                    print(f"(Could not read cache tensor shapes: {e})")
            print(f"Decode token (eos_id): {eos_id}")
            print(f"input_ids shape: {input_ids.shape}")
            try:
                decoded = self.tokenizer.decode(
                    input_ids[0].tolist(), skip_special_tokens=False
                )
                print(f"Decoded input_ids[0]: {decoded}")
            except Exception as e:
                print(f"(Could not decode input_ids: {e})")
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"cache_position: {cache_position}")
            print("use_cache: True")
            print("=== END DEBUG ===\n")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": static_cache,
            "cache_position": cache_position,
            "use_cache": True,
        }

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        """Generates text .
        Args:
            max_new_tokens (int): The maximum number of new tokens to generate.
            model (torch.nn.Module): The language model used for token generation.
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len), representing tokenized text.
            tokenizer: The tokenizer used to decode token IDs into text.
        """
        current_pos = self.seq_len

        for _ in range(max_new_tokens):
            logits = model(*inputs)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            next_token_logits = logits[:, current_pos - 1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Update input_ids and attention_mask
            inputs[0][:, current_pos] = next_token_id
            inputs[1][:, current_pos] = 1

            current_pos += 1

        valid_tokens = inputs[0][:, self.seq_len : current_pos].view(-1).tolist()
        answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        return answer

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        if self._variant in [
            ModelVariant.LLAMA_3_2_1B,
            ModelVariant.LLAMA_3_2_1B_INSTRUCT,
            ModelVariant.LLAMA_3_2_3B,
            ModelVariant.LLAMA_3_2_3B_INSTRUCT,
            ModelVariant.HUGGYLLAMA_7B,
        ]:
            return None

        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Llama model variant.

        Returns:
            The configuration object for the Llama model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
