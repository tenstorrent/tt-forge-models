# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 model loader implementation for text-only modeling.
"""

from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

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

    QWEN_3_5_27B       = "Qwen/Qwen3.5-27B"
    QWEN_3_5_35B_A3B   = "Qwen/Qwen3.5-35B-A3B"
    QWEN_3_5_122B_A10B = "Qwen/Qwen3.5-122B-A10B"
    QWEN_3_5_397B_A17B = "Qwen/Qwen3.5-397B-A17B"


class ModelLoader(ForgeModel):
    """Qwen 3.5 model loader implementation for text-only modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_27B),
        ),
        ModelVariant.QWEN_3_5_35B_A3B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_35B_A3B),
        ),
        ModelVariant.QWEN_3_5_122B_A10B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_122B_A10B),
        ),
        ModelVariant.QWEN_3_5_397B_A17B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.QWEN_3_5_397B_A17B),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B

    sample_text = "What is the capital of France?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override   # fixed: was "torch_dtype"
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, prompt: Optional[str] = None):
        """Load and return sample text inputs for the Qwen 3.5 model.

        Returns:
            dict: {
                "input_ids":      LongTensor [1, seq_len],
                "attention_mask": LongTensor [1, seq_len],
            }
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        text_prompt = prompt or self.sample_text

        messages = [{"role": "user", "content": text_prompt}]

        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        return self.tokenizer(
            formatted_text,
            return_tensors="pt",
            return_attention_mask=True,
        )

    def get_mesh_config(self, num_devices: int):
        """Get mesh configuration for tensor parallel execution.

        Args:
            num_devices: Number of devices to shard across.

        Returns:
            tuple: (mesh_shape, mesh_axis_names)
        """
        if num_devices == 32:  # Galaxy
            mesh_shape = (8, 4)
        else:
            mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Build tensor-parallel sharding spec for Qwen3.5 text model.

        Layer pattern repeating every 4 layers (16 times = 64 total):
            Layer i+0,i+1,i+2 : Qwen3_5GatedDeltaNet (linear_attn)
            Layer i+3          : Qwen3_5Attention      (self_attn)
        """
        shard_specs = {}

        for layer in model.model.layers:
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight]   = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            if hasattr(layer, "linear_attn"):
                la = layer.linear_attn
                shard_specs[la.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[la.in_proj_z.weight]   = ("model", "batch")
                shard_specs[la.conv1d.weight]      = ("model", "batch", None)
                shard_specs[la.out_proj.weight]    = ("batch", "model")

            elif hasattr(layer, "self_attn"):
                sa = layer.self_attn
                shard_specs[sa.q_proj.weight] = ("model", "batch")
                shard_specs[sa.k_proj.weight] = ("model", "batch")
                shard_specs[sa.v_proj.weight] = ("model", "batch")
                shard_specs[sa.o_proj.weight] = ("batch", "model")

        return shard_specs