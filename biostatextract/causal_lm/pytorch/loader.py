# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BioStatExtract causal LM model loader implementation.

``niwiyi/biostatextract`` is a LoRA adapter (PEFT) trained for biostatistics
extraction on top of ``unsloth/qwen2.5-7b-instruct-bnb-4bit`` (a 4-bit
quantized copy of ``Qwen/Qwen2.5-7B-Instruct``). The 4-bit base requires
bitsandbytes/CUDA, so we attach the adapter to the full-precision
``Qwen/Qwen2.5-7B-Instruct`` base instead and merge the LoRA weights to obtain
a plain ``Qwen2ForCausalLM`` that runs on CPU and Tenstorrent hardware.
"""


import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM, AutoConfig
from peft import PeftModel
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
    """Available BioStatExtract model variants for causal language modeling."""

    BIOSTATEXTRACT_7B_INSTRUCT = "7b_instruct"


class ModelLoader(ForgeModel):
    """BioStatExtract model loader implementation for causal language modeling tasks."""

    # Full-precision base the LoRA adapter is merged onto. The adapter's recorded
    # base (unsloth/qwen2.5-7b-instruct-bnb-4bit) is the 4-bit quantized variant
    # of this model and is not usable without bitsandbytes/CUDA.
    _BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BIOSTATEXTRACT_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="niwiyi/biostatextract",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BIOSTATEXTRACT_7B_INSTRUCT

    # Shared configuration parameters
    sample_text = (
        "Extract the sample size, mean, and standard deviation from the "
        "following clinical study result."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.seq_len = None

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
            model="BioStatExtract",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        The tokenizer (including the chat template and added tokens) is loaded
        from the adapter repository, which carries the task-specific vocabulary.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BioStatExtract model instance for this instance's variant.

        Loads the full-precision Qwen2.5-7B-Instruct base, attaches the
        ``niwiyi/biostatextract`` LoRA adapter, and merges it into the base
        weights to produce a standalone ``Qwen2ForCausalLM``.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its default dtype (float32).

        Returns:
            torch.nn.Module: The merged BioStatExtract model for causal language modeling.
        """
        adapter_id = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = Qwen2ForCausalLM.from_pretrained(
            self._BASE_MODEL_NAME, **model_kwargs
        )

        # Attach the pretrained LoRA adapter and fold it into the base weights.
        peft_model = PeftModel.from_pretrained(base_model, adapter_id)
        model = peft_model.merge_and_unload()
        model.eval()

        # merge_and_unload may leave residual float32 LoRA-derived tensors; ensure
        # the whole module matches the requested dtype.
        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BioStatExtract model.

        Args:
            dtype_override: Optional torch.dtype (only affects float tensors; ids stay int).
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
        """Load and return the configuration for the BioStatExtract base model.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(self._BASE_MODEL_NAME)
        return self.config
