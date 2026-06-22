# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Llama causal LM model loader implementation.
"""

import torch
from typing import Optional

# NOTE: `transformers` is intentionally NOT imported at module top level.
# This model pins transformers==4.53.3 (see requirements.txt) because its DeciLM
# remote code imports NEED_SETUP_CACHE_CLASSES_MAPPING, which was removed in
# later releases. The test runner installs that pin at test time and purges
# transformers from sys.modules. A top-level import would bind the Auto* classes
# to whatever transformers was loaded during pytest collection, leaving stale
# class objects whose in-memory code mismatches the pinned files on disk. So the
# Auto* classes are imported lazily inside the methods that use them.

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
    """Available NVIDIA Llama model variants for causal language modeling."""

    Nvidia_Llama_3_3_Nemotron_Super_49B_v1_5 = (
        "Nvidia_Llama_3_3_Nemotron_Super_49B_v1_5"
    )


class ModelLoader(ForgeModel):
    """NVIDIA Llama model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.Nvidia_Llama_3_3_Nemotron_Super_49B_v1_5: LLMModelConfig(
            pretrained_model_name="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Nvidia_Llama_3_3_Nemotron_Super_49B_v1_5

    sample_text = "Who are you?"

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
            model="Llama-Nemotron",
            variant=variant,
            group=ModelGroup.GENERALITY,
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
        # Lazy import so it binds to the pinned transformers (see module note).
        from transformers import AutoTokenizer

        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NVIDIA Llama model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The NVIDIA Llama model for causal language modeling.
        """
        # Lazy import so it binds to the pinned transformers (see module note).
        from transformers import AutoModelForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
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
        """Load and return sample inputs for the NVIDIA Llama model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length
        conversation = [{"role": "user", "content": self.sample_text}]
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
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
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads "
                f"across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}

        # Pin the embedding and final norm as replicated (sharded only on the
        # size-1 "batch" axis). This keeps the residual stream replicated, which
        # is what forces the compiler to insert the all-reduce after every
        # row-parallel o_proj/down_proj. Omitting these lets sharding
        # propagation infer a "model"-sharded residual stream and drop those
        # collectives, silently corrupting the result (PCC ~0.37).
        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.model.norm.weight] = ("batch",)
        shard_specs[model.lm_head.weight] = ("model", "batch")

        for layer in model.model.layers:
            # Llama-3.3-Nemotron-Super uses Neural Architecture Search, so some
            # blocks may replace attention or MLP with no-op modules. Guard each
            # access so we only shard the weights that exist.
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "up_proj"):
                # Column-parallel up/gate, row-parallel down.
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")

            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "q_proj"):
                # Column-parallel q/k/v (split by head), row-parallel o.
                shard_specs[attn.q_proj.weight] = ("model", "batch")
                shard_specs[attn.k_proj.weight] = ("model", "batch")
                shard_specs[attn.v_proj.weight] = ("model", "batch")
                shard_specs[attn.o_proj.weight] = ("batch", "model")

            # Layernorms are replicated. Attention-free NAS blocks have no
            # input_layernorm, so guard before annotating.
            if hasattr(layer, "input_layernorm"):
                shard_specs[layer.input_layernorm.weight] = ("batch",)
            if hasattr(layer, "post_attention_layernorm"):
                shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        # Lazy import so it binds to the pinned transformers (see module note).
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
