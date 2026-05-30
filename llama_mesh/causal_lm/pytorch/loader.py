# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaMA-Mesh model loader implementation for causal language modeling.

LLaMA-Mesh (Zhengyi/LLaMA-Mesh) is a Llama-3.1-8B model fine-tuned to generate
3D meshes (OBJ vertex/face text) from natural-language prompts. This loader
pulls the weights from the GGUF release (bartowski/LLaMA-Mesh-GGUF); transformers
dequantizes the GGUF checkpoint into a standard LlamaForCausalLM at load time.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import torch

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
    """Available LLaMA-Mesh model variants for causal LM."""

    LLAMA_MESH_8B = "8b"


class ModelLoader(ForgeModel):
    """LLaMA-Mesh model loader implementation for causal language modeling tasks.

    Weights are loaded from the GGUF release and dequantized to a regular
    LlamaForCausalLM by transformers.
    """

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LLAMA_MESH_8B: LLMModelConfig(
            pretrained_model_name="bartowski/LLaMA-Mesh-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_MESH_8B

    # GGUF quantization file to dequantize the weights from. Q4_K_M is a widely
    # supported K-quant that transformers can dequantize into fp32/bf16 weights.
    GGUF_FILE = "LLaMA-Mesh-Q4_K_M.gguf"

    # Sample prompt for mesh generation (LLaMA-Mesh is a conversational model).
    sample_text = "Create a 3D model of a simple chair."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to keep. If None, uses
                        the model's default (32).
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="LLaMA-Mesh",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the GGUF release.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )

        # Llama tokenizers ship without a pad token; reuse eos for padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, num_layers=None, **kwargs):
        """Load and return the LLaMA-Mesh model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to cast the model to. If not
                            provided, the dequantized GGUF default is used.
            num_layers: Optional number of hidden layers to keep. If None, uses
                        the model's default.

        Returns:
            torch.nn.Module: The LlamaForCausalLM model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Optionally truncate layers (useful for quick bringup / memory limits).
        layers_to_keep = num_layers if num_layers is not None else self.num_layers
        if layers_to_keep is not None:
            model.model.layers = model.model.layers[:layers_to_keep]
            model.config.num_hidden_layers = layers_to_keep

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the LLaMA-Mesh model.

        Args:
            dtype_override: Optional torch.dtype to cast the inputs to.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask to a fixed length for static shapes.
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
