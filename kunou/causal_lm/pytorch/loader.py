# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
14B-Qwen2.5-Kunou-v1 Causal LM model loader implementation.

Kunou-v1 is a Qwen2.5-14B fine-tune. This loader brings up the model from the
GGUF-quantized release ``mradermacher/14B-Qwen2.5-Kunou-v1-GGUF``: transformers
dequantizes the GGUF weights back into a standard ``Qwen2ForCausalLM`` torch
module (both the model and the tokenizer are reconstructed from the GGUF file).
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
    """Available 14B-Qwen2.5-Kunou-v1 variants for causal language modeling."""

    KUNOU_14B_V1_Q4_K_M = "14B_v1_Q4_K_M"


class ModelLoader(ForgeModel):
    """14B-Qwen2.5-Kunou-v1 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.KUNOU_14B_V1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/14B-Qwen2.5-Kunou-v1-GGUF",
            max_length=128,
        ),
    }

    # GGUF file inside the repo to dequantize for each variant.
    _GGUF_FILES = {
        ModelVariant.KUNOU_14B_V1_Q4_K_M: "14B-Qwen2.5-Kunou-v1.Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.KUNOU_14B_V1_Q4_K_M

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the
                        model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="14B-Qwen2.5-Kunou-v1",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kunou-v1 model instance for this instance's variant.

        The weights are dequantized from the GGUF release into a standard
        Qwen2ForCausalLM module via transformers' ``gguf_file`` support.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its default (float32).

        Returns:
            torch.nn.Module: The Kunou-v1 model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        # NOTE on memory: transformers dequantizes the GGUF weights into a full
        # fp32 state dict (~56GB for 14B). Passing torch_dtype here would make it
        # hold that fp32 dict AND a second cast (bf16) copy of the whole model at
        # the same time (~84GB peak), which OOMs a 90GB host. Instead we load in
        # fp32 (the dequantized tensors become the params in place, ~56GB peak)
        # and then cast to the requested dtype one tensor at a time below, so the
        # fp32 and target copies never coexist for the whole model.
        model_kwargs = {"gguf_file": gguf_file}
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        if dtype_override is not None:
            # In-place, per-tensor cast to keep host memory peak ~constant.
            for param in model.parameters():
                if param.data.is_floating_point():
                    param.data = param.data.to(dtype_override)
            for buffer in model.buffers():
                if buffer is not None and buffer.is_floating_point():
                    buffer.data = buffer.data.to(dtype_override)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Kunou-v1 model.

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the Kunou-v1 model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
