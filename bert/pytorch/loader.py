# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT model loader implementation for question answering
"""
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """BERT model loader implementation for question answering tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        "base": {
            "pretrained_model_name": "phiyodr/bert-base-finetuned-squad2",
            "max_length": 256,
            "description": "BERT-base fine-tuned on SQuAD v2",
        },
        "large": {
            "pretrained_model_name": "phiyodr/bert-large-finetuned-squad2",
            "max_length": 256,
            "description": "BERT-large fine-tuned on SQuAD v2",
        },
    }

    # Default variant to use
    DEFAULT_VARIANT = "large"

    # Shared configuration parameters
    context = 'Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. "The prophet and founding hero of modern archaeology", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. '
    question = "What discipline did Winkelmann create?"

    # Tokenizer shared across instances
    tokenizer = None

    def load_model(self, dtype_override=None):
        """Load and return the BERT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BERT model instance for question answering.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config["pretrained_model_name"]

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {"padding_side": "left"}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Initialize the tokenizer if not already done
        if ModelLoader.tokenizer is None:
            ModelLoader.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name, **tokenizer_kwargs
            )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BERT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if ModelLoader.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config["max_length"]

        # Create tokenized inputs
        inputs = ModelLoader.tokenizer.encode_plus(
            self.question,
            self.context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded answer text
        """
        # Ensure tokenizer is initialized
        if ModelLoader.tokenizer is None:
            self.load_model()

        if inputs is None:
            inputs = self.load_inputs()

        response_start = torch.argmax(outputs.start_logits)
        response_end = torch.argmax(outputs.end_logits) + 1
        response_tokens = inputs.input_ids[0, response_start:response_end]

        return ModelLoader.tokenizer.decode(response_tokens)
