# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-R1-Distill-Llama-8B (GGUF) causal LM model loader implementation.

The weights are published as GGUF quantized files. transformers loads and
dequantizes them into a standard ``LlamaForCausalLM`` via the ``gguf_file``
argument, so the resulting module layout matches the regular Llama loaders.
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
    """Available DeepSeek-R1-Distill-Llama GGUF variants."""

    DEEPSEEK_LLAMA_8B_Q4_K_M = "DeepSeek-R1-Distill-Llama-8B-GGUF-Q4_K_M"


class ModelLoader(ForgeModel):
    """DeepSeek-R1-Distill-Llama-8B GGUF loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_LLAMA_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="matrixportalx/DeepSeek-R1-Distill-Llama-8B-GGUF",
            max_length=256,
        )
    }

    # GGUF file inside the repo for each variant. transformers reads/dequantizes
    # this file via the ``gguf_file`` argument to from_pretrained.
    _GGUF_FILES = {
        ModelVariant.DEEPSEEK_LLAMA_8B_Q4_K_M: "deepseek-r1-distill-llama-8b-q4_k_m.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_LLAMA_8B_Q4_K_M

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

    @property
    def _gguf_file(self) -> str:
        return self._GGUF_FILES[self._variant]

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
            model="DeepSeek Llama GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype (unused for the tokenizer, kept
                            for signature parity with the model loader).

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepSeek Llama GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, defaults to bfloat16.

        Returns:
            torch.nn.Module: The DeepSeek Llama model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": self._gguf_file}
        model_kwargs["torch_dtype"] = (
            dtype_override if dtype_override is not None else torch.bfloat16
        )
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DeepSeek Llama GGUF model.

        Args:
            dtype_override: Optional torch.dtype (unused, inputs are integer ids).
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Use natural-length tokenization (no fixed-length zero padding) so the
        # PCC comparison on device is not skewed by padding tokens.
        try:
            conversation = [{"role": "user", "content": self.sample_text}]
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = self.sample_text

        inputs = self.tokenizer(prompt, return_tensors="pt")

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
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
        )
        return self.config
