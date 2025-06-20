# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ALBERT model loader implementation for masked language modeling.
"""
import torch
from transformers import AlbertForMaskedLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)


class ModelLoader(ForgeModel):
    """ALBERT model loader implementation for masked language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        "albert-base-v2": LLMModelConfig(
            pretrained_model_name="albert/albert-base-v2",
            max_length=128,
        ),
        "albert-large-v2": LLMModelConfig(
            pretrained_model_name="albert/albert-large-v2",
            max_length=128,
        ),
        "albert-xlarge-v2": LLMModelConfig(
            pretrained_model_name="albert/albert-xlarge-v2",
            max_length=128,
        ),
        "albert-xxlarge-v2": LLMModelConfig(
            pretrained_model_name="albert/albert-xxlarge-v2",
            max_length=128,  # Added default max length
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = "albert-base-v2"

    # Shared configuration parameters
    sample_text = "The capital of [MASK] is Paris."

    # Tokenizer shared across instances
    tokenizer = None

    @classmethod
    def get_model_info(cls, variant=None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional string specifying which variant to get info for.

        Returns:
            ModelInfo: Information about the model and variant
        """
        variant = cls._validate_variant(variant)

        return ModelInfo(
            model="albert_v2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the ALBERT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ALBERT model instance for masked language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AlbertForMaskedLM.from_pretrained(pretrained_model_name, **model_kwargs)

        # Initialize the tokenizer if not already done
        # Note: tokenizer is shared across all instances
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        if ModelLoader.tokenizer is None:
            ModelLoader.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name, **tokenizer_kwargs
            )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ALBERT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if ModelLoader.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        # Create tokenized inputs for the masked language modeling task
        inputs = ModelLoader.tokenizer(self.sample_text, return_tensors="pt")
        return inputs

    @classmethod
    def decode_output(cls, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded prediction for the masked token
        """
        # Ensure tokenizer is initialized
        if ModelLoader.tokenizer is None:
            self.load_model()

        if inputs is None:
            inputs = self.load_inputs()

        # Get the prediction for the masked token
        logits = outputs.logits
        mask_token_index = (inputs.input_ids == ModelLoader.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = ModelLoader.tokenizer.decode(predicted_token_id)

        return predicted_tokens
