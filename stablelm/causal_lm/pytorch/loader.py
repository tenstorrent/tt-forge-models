# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
StableLM model loader implementation for causal language modeling.

StableLM base-alpha models use the GPT-NeoX architecture:
    model.gpt_neox.layers[i].attention.query_key_value  (fused QKV)
    model.gpt_neox.layers[i].attention.dense            (output projection)
    model.gpt_neox.layers[i].mlp.dense_h_to_4h         (MLP expand)
    model.gpt_neox.layers[i].mlp.dense_4h_to_h         (MLP contract)
"""
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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


class ModelVariant(StrEnum):
    """Available StableLM model variants."""

    _3B_ALPHA = "3B_Alpha"
    _7B_ALPHA = "7B_Alpha"


class ModelLoader(ForgeModel):
    """StableLM model loader for causal language modeling (single- and multi-chip)."""

    _VARIANTS = {
        ModelVariant._3B_ALPHA: LLMModelConfig(
            pretrained_model_name="stabilityai/stablelm-base-alpha-3b",
            max_length=256,
        ),
        ModelVariant._7B_ALPHA: LLMModelConfig(
            pretrained_model_name="stabilityai/stablelm-base-alpha-7b",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant._3B_ALPHA
    sample_text = "What's your mood today?"

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

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="StableLM",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load and cache the tokenizer for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the StableLM model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use float32.

        Returns:
            torch.nn.Module: The StableLM model in eval mode.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the StableLM model.

        Args:
            dtype_override: Unused for integer token inputs; kept for API consistency.
            batch_size: Number of input sequences.

        Returns:
            dict: input_ids and attention_mask tensors.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        tokenized = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        inputs = {
            "input_ids": tokenized.input_ids.repeat_interleave(batch_size, dim=0),
            "attention_mask": tokenized.attention_mask.repeat_interleave(batch_size, dim=0),
        }
        return inputs

    def _get_config(self):
        """Load and cache the HuggingFace config for the current variant."""
        if not hasattr(self, "_cached_config") or self._cached_config is None:
            self._cached_config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self._cached_config

    def get_mesh_config(self, num_devices: int):
        """Return mesh topology for multi-chip tensor parallelism.

        StableLM base-alpha head counts:
            3B → 32 heads  (fits: 2, 4, 8, 16, 32)
            7B → 32 heads  (fits: 2, 4, 8, 16, 32)

        Args:
            num_devices: Total number of available TT devices.

        Returns:
            (mesh_shape, mesh_axis_names): e.g. ((1, 2), ("batch", "model"))
        """
        if num_devices == 1:
            return (1, 1), ("batch", "model")

        config = self._get_config()
        num_heads = config.num_attention_heads

        if num_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif num_devices % 2 == 0 and num_heads % (num_devices // 2) == 0:
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {num_heads} attention heads across "
                f"{num_devices} devices. Supported counts: divisors of {num_heads}."
            )

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Define per-tensor sharding for tensor parallelism.

        Applies Megatron-style column/row parallelism:
          - Fused QKV and MLP expand: shard on output dim (column-parallel)
          - Output projection and MLP contract: shard on input dim (row-parallel)

        Args:
            model: Loaded AutoModelForCausalLM (GPTNeoXForCausalLM) instance.

        Returns:
            Dict mapping weight tensors to shard-spec tuples.
        """
        shard_specs = {}

        for layer in model.gpt_neox.layers:
            attn = layer.attention
            # Fused QKV — column-parallel (shard output/row 0)
            shard_specs[attn.query_key_value.weight] = ("model", None)
            # Output projection — row-parallel (shard input/col 1)
            shard_specs[attn.dense.weight] = (None, "model")

            mlp = layer.mlp
            # MLP expand (4× hidden) — column-parallel
            shard_specs[mlp.dense_h_to_4h.weight] = ("model", None)
            # MLP contract (back to hidden) — row-parallel
            shard_specs[mlp.dense_4h_to_h.weight] = (None, "model")

        return shard_specs
