# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 model loader implementation for text-only modeling.
"""

import types
from typing import Optional

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Qwen 3.5 text-only model variants."""

    QWEN_3_5_27B = "27B"
    QWEN_3_5_35B_A3B = "35B_A3B"
    QWEN_3_5_122B_A10B = "122B_A10B"
    QWEN_3_5_397B_A17B = "397B_A17B"


class ModelLoader(ForgeModel):
    """Qwen 3.5 model loader implementation for text-only modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-27B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-35B-A3B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_122B_A10B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-122B-A10B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_397B_A17B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-397B-A17B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B

    sample_text = (
        "Explain the key differences between transformer-based large language "
        "models and traditional recurrent neural networks, focusing on attention "
        "mechanisms, parallelism, and long-range dependency handling."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="qwen_3_5_text",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.5 text-only causal LM instance.

        Args:
            dtype_override: Optional torch.dtype to override model default dtype.

        Returns:
            torch.nn.Module: The Qwen 3.5 causal LM instance.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self, dtype_override=None, prompt: Optional[str] = None, batch_size=1
    ):
        """Load and return sample text inputs for the Qwen 3.5 model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        text_prompt = prompt or self.sample_text
        formatted_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self._variant_config.max_length,
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shapes = {
            2: (1, 2),
            4: (2, 2),
            8: (2, 4),
            16: (4, 4),
            32: (8, 4),  # Galaxy
        }
        mesh_shape = mesh_shapes.get(num_devices, (1, num_devices))
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}

        for layer in model.model.layers:
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "self_attn"):
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("model", "batch")
                shard_specs[sa.k_proj.weight] = ("model", "batch")
                shard_specs[sa.v_proj.weight] = ("model", "batch")
                shard_specs[sa.o_proj.weight] = ("batch", "model")

            elif hasattr(layer, "linear_attn"):
                la = layer.linear_attn
                shard_specs[la.in_proj_qkv.weight] = ("model", "batch")
                if hasattr(la, "conv1d"):
                    shard_specs[la.conv1d.weight] = ("model", None, None)
                shard_specs[la.in_proj_z.weight] = ("model", "batch")
                shard_specs[la.in_proj_a.weight] = ("model", "batch")
                shard_specs[la.in_proj_b.weight] = ("model", "batch")
                shard_specs[la.out_proj.weight] = ("batch", "model")

        shard_specs[model.model.embed_tokens.weight] = ("model", "batch")
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs
