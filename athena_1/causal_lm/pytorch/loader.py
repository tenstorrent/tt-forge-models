# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Athena-1 GGUF Causal LM model loader implementation

Athena-1-3B is a Qwen2-architecture causal language model. This loader pulls a
quantized GGUF checkpoint from the mradermacher imatrix repository and lets
transformers de-quantize it into a standard ``Qwen2ForCausalLM`` instance.
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
    """Available Athena-1 GGUF model variants for causal language modeling."""

    ATHENA_1_3B_I1_Q4_K_M = "3B_i1_Q4_K_M"


class ModelLoader(ForgeModel):
    """Athena-1 GGUF model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.ATHENA_1_3B_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Athena-1-3B-i1-GGUF",
            max_length=128,
        ),
    }

    # GGUF checkpoint file (within the repo) backing each variant.
    _GGUF_FILES = {
        ModelVariant.ATHENA_1_3B_I1_Q4_K_M: "Athena-1-3B.i1-Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ATHENA_1_3B_I1_Q4_K_M

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Not supported for GGUF loaders — the weights are baked
                        into the quantized checkpoint and cannot be truncated.
        """
        super().__init__(variant)
        if num_layers is not None:
            raise ValueError(
                "num_layers override requested but ModelLoader does not support it "
                "(GGUF checkpoints carry fixed weights for all layers)."
            )
        self.tokenizer = None
        self.config = None
        self.seq_len = None

    @property
    def _gguf_file(self) -> str:
        """Return the GGUF checkpoint filename for the current variant."""
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
        return ModelInfo(
            model="Athena-1",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Athena-1 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its default dtype.

        Returns:
            torch.nn.Module: The Athena-1 (Qwen2) model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Athena-1 model with this instance's variant settings.

        Args:
            dtype_override: Unused for token inputs; kept for interface parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
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
        """Load and return the configuration for the Athena-1 model variant.

        Returns:
            The configuration object for the Athena-1 model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )

        return self.config
