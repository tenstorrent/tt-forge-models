# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.2 90B Vision Instruct model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

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
    """Available Llama 3.2 model variants for causal language modeling."""

    LLAMA_3_2_90B_VISION_INSTRUCT = "Llama-3.2-90B-Vision-Instruct"


class ModelLoader(ForgeModel):
    """Llama 3.2 90B Vision Instruct model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_90B_VISION_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-90B-Vision-Instruct",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_90B_VISION_INSTRUCT

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
            model="Llama 3.2",
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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llama 3.2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Llama 3.2 model for causal language modeling.
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
        """Load and return sample inputs for the Llama 3.2 model with this instance's variant settings.

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

    def _text_config(self):
        """MllamaConfig nests heads on ``text_config``; CausalLM may expose them on the root."""
        return getattr(self.config, "text_config", self.config)

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        text_cfg = self._text_config()
        attn_heads = text_cfg.num_attention_heads
        kv_heads = getattr(text_cfg, "num_key_value_heads", attn_heads)
        model_axis = mesh_shape[1]
        if attn_heads % model_axis != 0:
            raise ValueError(
                f"Cannot evenly distribute {attn_heads} attention heads "
                f"across model axis size {model_axis}"
            )
        if kv_heads % model_axis != 0:
            raise ValueError(
                f"Cannot evenly distribute {kv_heads} KV heads "
                f"across model axis size {model_axis}"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP for Mllama text layers.

        The decoder interleaves ``MllamaSelfAttentionDecoderLayer`` (self_attn)
        and ``MllamaCrossAttentionDecoderLayer`` (cross_attn). Both expose the
        same q/k/v/o shapes: q/o are 8192 (64 heads), k/v are 1024 (8 KV
        heads). 8 KV heads divide the model axis on 4-chip and Galaxy, so
        column-parallel q/k/v and row-parallel o is valid. MLP is gate/up
        column, down row. Cross-attn ``q_norm`` / ``k_norm`` stay replicated.
        """
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            attn = layer.self_attn if hasattr(layer, "self_attn") else layer.cross_attn
            shard_specs[attn.q_proj.weight] = ("model", "batch")
            shard_specs[attn.k_proj.weight] = ("model", "batch")
            shard_specs[attn.v_proj.weight] = ("model", "batch")
            shard_specs[attn.o_proj.weight] = ("batch", "model")

        shard_specs[model.model.embed_tokens.weight] = ("model", "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
