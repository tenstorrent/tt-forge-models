# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llasa model loader implementation for text-to-speech via causal language modeling.

Llasa (HKUSTAudio/Llasa-8B) is a Llama-3.1-8B finetune whose vocabulary is
extended with speech (XCodec2) tokens. It generates speech tokens autoregressively
the same way a causal LM generates text, so it is brought up here as a single
``LlamaForCausalLM`` forward pass. The downstream XCodec2 vocoder that turns the
generated speech tokens into a waveform is a separate model and is out of scope
for this loader.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    """Available Llasa model variants for TTS causal LM."""

    LLASA_8B = "8b"


class ModelLoader(ForgeModel):
    """Llasa model loader implementation for text-to-speech causal LM tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LLASA_8B: LLMModelConfig(
            pretrained_model_name="HKUSTAudio/Llasa-8B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLASA_8B

    # Sample text to synthesize (wrapped in the Llasa TTS chat template below).
    sample_text = "Hello, this is a test of the Llasa text to speech model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None

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
            model="Llasa",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llasa model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its checkpoint dtype.

        Returns:
            torch.nn.Module: The Llasa LlamaForCausalLM model instance.
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
        self.model = model
        self.config = model.config

        return model

    def _build_prompt_ids(self):
        """Build input ids for the Llasa TTS prompt using its chat template.

        The prompt follows the model card: the text to synthesize is wrapped in
        ``<|TEXT_UNDERSTANDING_START|> ... <|TEXT_UNDERSTANDING_END|>`` and the
        assistant turn is primed with ``<|SPEECH_GENERATION_START|>`` so the model
        continues by emitting speech tokens.
        """
        formatted_text = (
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>"
            + self.sample_text
            + "<|TEXT_UNDERSTANDING_END|>"
        )
        chat = [
            {"role": "user", "content": formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]
        encoded = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        )
        # Depending on the transformers version / tokenizer, this returns either a
        # tensor of ids or a BatchEncoding/dict containing "input_ids".
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids
        if isinstance(encoded, dict):
            return encoded["input_ids"]
        return encoded

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llasa model.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        input_ids = self._build_prompt_ids()
        attention_mask = torch.ones_like(input_ids)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad to a fixed static length so the graph has static shapes on device.
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_config(self):
        """Load and return the configuration for the Llasa model variant.

        Returns:
            The configuration object for the Llasa model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
