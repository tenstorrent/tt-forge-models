# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SeamlessM4T v2 model loader implementation for speech/text translation.

The model under test is the text decoder submodule of ``SeamlessM4Tv2Model``
(``facebook/seamless-m4t-v2-large``). The decoder is a 24-layer encoder-decoder
transformer that cross-attends to encoder hidden states; it is the compute-heavy
text-generation core shared by every SeamlessM4T v2 task (S2T, T2T, S2S, T2S).
"""
import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SeamlessM4T v2 model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """SeamlessM4T v2 model loader for speech/text translation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/seamless-m4t-v2-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE

    # Sequence lengths for the sample inputs.
    ENCODER_SEQ_LEN = 16
    DECODER_SEQ_LEN = 8

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.full_model = None

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
            model="SeamlessM4Tv2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor_and_config(self):
        """Load processor and config for the current variant.

        Returns:
            tuple: (processor, config) instances
        """
        from transformers import AutoProcessor, SeamlessM4Tv2Config

        model_name = self._variant_config.pretrained_model_name

        # Load config and processor
        self.config = SeamlessM4Tv2Config.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        return self.processor, self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SeamlessM4T v2 text decoder submodule.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its default (float32).

        Returns:
            torch.nn.Module: The SeamlessM4T v2 text decoder submodule.
        """
        from transformers import SeamlessM4Tv2Model

        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Ensure processor and config are loaded
        if self.processor is None or self.config is None:
            self._load_processor_and_config()

        # Load directly in the requested dtype when provided to keep memory low.
        if dtype_override is not None:
            kwargs.setdefault("dtype", dtype_override)

        # Load the full model so the text decoder carries pretrained weights.
        self.full_model = SeamlessM4Tv2Model.from_pretrained(
            model_name, config=self.config, **kwargs
        )

        if dtype_override is not None:
            self.full_model = self.full_model.to(dtype_override)

        # Return the text decoder submodule only.
        return self.full_model.text_decoder

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SeamlessM4T v2 text decoder.

        Args:
            dtype_override: Optional torch.dtype to cast float inputs to.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Keyword arguments that can be fed to the text decoder.
        """
        # Ensure config is available for the hidden size / token ids.
        if self.config is None:
            self._load_processor_and_config()

        hidden_size = self.config.hidden_size
        bos_token_id = self.config.bos_token_id

        # Deterministic encoder hidden states standing in for the speech/text
        # encoder output the decoder cross-attends to.
        generator = torch.Generator().manual_seed(0)
        encoder_hidden_states = torch.randn(
            batch_size,
            self.ENCODER_SEQ_LEN,
            hidden_size,
            generator=generator,
        )

        # Decoder input token ids beginning with the BOS token.
        decoder_input_ids = torch.full(
            (batch_size, self.DECODER_SEQ_LEN), bos_token_id, dtype=torch.long
        )

        if dtype_override is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        # use_cache=False keeps the traced graph to a single tensor output.
        arguments = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "use_cache": False,
        }

        return arguments

    def decode_output(self, outputs):
        """Helper to summarise decoder outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass of the text decoder.

        Returns:
            str: Summary of the decoder output.
        """
        if self.full_model is None:
            raise RuntimeError("Model must be loaded before decoding output")

        last_hidden_state = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

        # Project to vocabulary logits with the model's lm_head for a quick read.
        logits = self.full_model.lm_head(last_hidden_state)
        predicted_ids = torch.argmax(logits, dim=-1)

        if self.processor is None:
            self._load_processor_and_config()
        decoded_text = self.processor.tokenizer.decode(
            predicted_ids[0], skip_special_tokens=True
        )

        return f"""
        SeamlessM4T v2 Output:
          - Decoded text: "{decoded_text}"
          - Hidden state shape: {last_hidden_state.shape}
          - Predicted token IDs: {predicted_ids[0].tolist()}
        """
