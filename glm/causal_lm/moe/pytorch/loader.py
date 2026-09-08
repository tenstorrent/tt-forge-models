# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash MoE model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from .....base import ForgeModel
from .....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GLM-4.7-Flash MoE model variants for causal language modeling."""

    GLM_4_7_FLASH = "GLM-4.7-Flash"


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash MoE model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH: LLMModelConfig(
            pretrained_model_name="zai-org/GLM-4.7-Flash",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH

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
            model="GLM-4.7-Flash MoE",
            variant=variant,
            group=ModelGroup.GENERALITY,
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

        # GLM's tokenizer may ship without a pad token; reuse EOS so padding works.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GLM-4.7-Flash MoE model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GLM-4.7-Flash MoE model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
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
        """Load and return sample inputs for the GLM-4.7-Flash MoE model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        if self.tokenizer.chat_template is not None:
            conversation = [{"role": "user", "content": self.sample_text}]
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self.sample_text
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
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            # GLM-4.7-Flash uses Multi-head Latent Attention (MLA): query and
            # key/value are recovered from low-rank latents via q_b_proj /
            # kv_b_proj, which are column-sharded over heads; o_proj is
            # row-sharded to match. The low-rank down-projections (q_a_proj /
            # kv_a_proj_with_mqa) produce a shared latent every device needs in
            # full, so they stay replicated.
            sa = layer.self_attn
            if hasattr(sa, "q_proj"):
                shard_specs[sa.q_proj.weight] = ("model", "batch")
            else:
                shard_specs[sa.q_b_proj.weight] = ("model", "batch")
            shard_specs[sa.kv_b_proj.weight] = ("model", "batch")
            shard_specs[sa.o_proj.weight] = ("batch", "model")

            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                # Sparse MoE layer: experts are stored as fused 3D tensors on
                # Glm4MoeLiteNaiveMoe (gate_up_proj / down_proj) and whole
                # experts are distributed across devices (expert-parallel). The
                # router (gate) must see all experts, so it stays replicated.
                shard_specs[mlp.experts.gate_up_proj] = ("model", None, None)
                shard_specs[mlp.experts.down_proj] = ("model", None, None)

                shared = getattr(mlp, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.gate_proj.weight] = ("model", "batch")
                    shard_specs[shared.up_proj.weight] = ("model", "batch")
                    shard_specs[shared.down_proj.weight] = ("batch", "model")
            else:
                # Dense MLP (first layer, before MoE kicks in).
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
