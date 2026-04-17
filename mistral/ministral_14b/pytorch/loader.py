# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral 14B model loader implementation.
"""

from typing import Optional

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
    """Available Ministral 14B model variants."""

    MINISTRAL_14B_BASE_2512 = "mistralai/Ministral-3-14B-Base-2512"
    MINISTRAL_14B_INSTRUCT_2512_BF16 = "mistralai/Ministral-3-14B-Instruct-2512-BF16"


class ModelLoader(ForgeModel):
    """Ministral 14B model loader implementation."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_14B_BASE_2512: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MINISTRAL_14B_BASE_2512),
        ),
        ModelVariant.MINISTRAL_14B_INSTRUCT_2512_BF16: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.MINISTRAL_14B_INSTRUCT_2512_BF16),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_14B_BASE_2512

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ministral_14b",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ministral 14B model instance.

        Uses Ministral3ForCausalLM (text-only) instead of
        Mistral3ForConditionalGeneration to avoid Pixtral vision model
        compilation issues with torch.compile.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Ministral 14B model instance.
        """
        from transformers import Ministral3ForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = Ministral3ForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load and return sample inputs for the Ministral 14B model.

        Uses text-only inputs to avoid Pixtral vision model dynamic shape
        issues with torch.compile (see pixtral loader for reference).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        import torch

        # Text-only input IDs for "What do you see in this image?"
        inputs = {
            "input_ids": torch.tensor(
                [[1, 7493, 1653, 1636, 3219, 1294, 1593, 3937, 1063]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long
            ),
        }

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution."""
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution."""
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
