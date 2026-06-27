# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llasa model loader implementation for text-to-speech.

Llasa is a LLaMA-based text-to-speech model: architecturally it is a
``LlamaForCausalLM`` decoder fine-tuned from Llama-3.1-8B-Instruct, with the
vocabulary extended by an XCodec2 speech codebook so that audio is produced as
an autoregressive continuation of speech tokens. The HuggingFace checkpoint is
a single causal-LM forward pass (the XCodec2 acoustic decoder that turns the
generated speech tokens back into a waveform is a separate model and is not part
of this loader).
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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Llasa model variants."""

    LLASA_8B = "8b"


class ModelLoader(ForgeModel):
    """Llasa model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LLASA_8B: LLMModelConfig(
            pretrained_model_name="HKUSTAudio/Llasa-8B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLASA_8B

    # Text the model is asked to synthesize into speech tokens.
    sample_text = "Hello, this is a text to speech test."

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
        self.model = None

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

        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Llasa model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the checkpoint dtype (bfloat16) is used.

        Returns:
            torch.nn.Module: The Llasa causal-LM model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        # Disable the KV cache for this single-forward-pass bringup (the common
        # convention across the causal-LM loaders here). Without it the model
        # also returns ``past_key_values``; those bf16 KV tensors are internal
        # intermediates never consumed in a one-shot forward, yet the runner's
        # min-over-output-leaves PCC includes them and their drift (~0.986)
        # masks the actual logits PCC (~0.9997).
        model_kwargs = {"use_cache": False}
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
        """Build Llasa TTS prompt token ids via the chat template.

        The prompt frames the input text between the speech-codec understanding
        markers and ends with ``<|SPEECH_GENERATION_START|>`` so the model
        continues by emitting speech tokens.
        """
        chat = [
            {
                "role": "user",
                "content": "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>"
                + self.sample_text
                + "<|TEXT_UNDERSTANDING_END|>",
            },
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]
        enc = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=True,
        )
        return enc["input_ids"], enc["attention_mask"]

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llasa model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the causal-LM
                  forward pass — the natural Llasa TTS prompt (no padding).

        Note:
            Unlike the autoregressive-generation loaders, this single-forward
            bringup feeds the prompt at its natural length and does NOT append
            generation-room padding. Padding positions are masked
            (attention_mask=0) and their logits are never consumed, but they are
            still computed; comparing those dead positions against the CPU
            reference injected ~0.04 of spurious bf16 noise into the full-tensor
            PCC (0.96 padded vs 0.9997 unpadded) without reflecting any real
            numerical error at the prediction position.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        input_ids, attention_mask = self._build_prompt_ids()

        # Replicate tensors for batch size
        input_ids = input_ids.repeat_interleave(batch_size, dim=0)
        attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)

        self.seq_len = input_ids.shape[1]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        """Decode the next predicted speech token id from forward-pass logits.

        Args:
            outputs: Output of the model forward pass (logits or a ModelOutput).

        Returns:
            int: The greedily predicted next speech-token id at the prompt end.
        """
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        next_token_logits = logits[:, self.seq_len - 1, :]
        return int(torch.argmax(next_token_logits, dim=-1)[0])

    def load_config(self):
        """Load and return the configuration for the Llasa model variant.

        Returns:
            The configuration object for the Llasa model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
