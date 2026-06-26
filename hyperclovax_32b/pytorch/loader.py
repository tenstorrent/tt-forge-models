# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HyperCLOVA X SEED Think causal LM model loader implementation.
"""

import torch
from typing import Optional

# NOTE: `transformers` is intentionally NOT imported at module top level.
# HyperCLOVA X SEED Think gained native support in transformers == 4.52.4 (see
# requirements.txt; model_type "hyperclovax"). The test runner upgrades
# transformers at test time and purges it from sys.modules. A top-level import
# would bind the Auto* classes to whatever transformers was loaded during pytest
# collection, leaving stale class objects whose in-memory code mismatches the
# pinned files on disk. So the Auto* classes are imported lazily in the methods.

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
    """Available HyperCLOVA X model variants for causal language modeling."""

    HyperCLOVAX_SEED_Think_32B = "HyperCLOVAX_SEED_Think_32B"


class ModelLoader(ForgeModel):
    """HyperCLOVA X model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HyperCLOVAX_SEED_Think_32B: LLMModelConfig(
            pretrained_model_name="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HyperCLOVAX_SEED_Think_32B

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
            model="HyperCLOVAX-SEED-Think",
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
        """Load and return the HyperCLOVA X model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The HyperCLOVA X model for causal language modeling.
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
        """Load and return sample inputs for the HyperCLOVA X model with this instance's variant settings.

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
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Tensor-parallel shard spec for the HyperCLOVA X SEED Think model."""
        shard_specs = {}

        inner = model.model
        language_model = inner.language_model
        lm = language_model.model

        # Keep the residual stream replicated (embedding + final norm pinned to
        # the size-1 "batch" axis) so the compiler inserts the all-reduces after
        # each row-parallel o_proj/down_proj instead of inferring a
        # "model"-sharded residual and dropping them (see nvidia/llama).
        shard_specs[lm.embed_tokens.weight] = (None, "batch")
        shard_specs[lm.norm.weight] = ("batch",)

        for layer in lm.layers:
            # Column-parallel q/k/v (GQA: k/v are smaller), row-parallel o.
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            # Column-parallel gate/up, row-parallel down.
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            # Layernorms replicated.
            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        shard_specs[language_model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        # Lazy import so it binds to the pinned transformers (see module note).
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
