# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SeamlessM4T v2 model loader implementation for speech-to-text translation.

Brings up the text decoder of ``facebook/seamless-m4t-v2-large`` as a single
forward pass on Tenstorrent hardware. The speech encoder is run on host to
produce ``encoder_hidden_states``; the text decoder is the module returned for
compilation / device execution.

Audio inputs are synthesized deterministically with numpy (a fixed sum of
tones), so the loader needs no ``torchaudio`` / network access. This is
deliberate: ``torchaudio`` pinned to the v1 SeamlessM4T requirement (2.9.0)
force-downgrades torch off the 2.10 stack the torch-xla device path needs, and
the prebuilt ``libtorchaudio.so`` fails to load in this environment.
"""
import numpy as np
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
    """SeamlessM4T v2 loader for speech-to-text translation (text-decoder bringup)."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/seamless-m4t-v2-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE

    # Sampling rate the SeamlessM4T feature extractor expects.
    SAMPLING_RATE = 16_000

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

        self.config = SeamlessM4Tv2Config.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        return self.processor, self.config

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the SeamlessM4T v2 text decoder submodule.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. If not provided, the model's default dtype is used.

        Returns:
            torch.nn.Module: The SeamlessM4T v2 text decoder submodule.
        """
        from transformers import SeamlessM4Tv2Model

        model_name = self._variant_config.pretrained_model_name

        if self.processor is None or self.config is None:
            self._load_processor_and_config()

        self.full_model = SeamlessM4Tv2Model.from_pretrained(
            model_name, config=self.config, **kwargs
        )
        self.full_model = self.full_model.eval()

        if dtype_override is not None:
            self.full_model = self.full_model.to(dtype_override)

        # Return the text decoder submodule (the part brought up on device).
        return self.full_model.text_decoder

    def _synthesize_audio(self) -> np.ndarray:
        """Return a deterministic 2-second 16 kHz mono waveform.

        Uses a fixed sum of sine tones so inputs are reproducible across runs
        without needing torchaudio or network access.
        """
        duration_s = 2.0
        t = np.arange(int(duration_s * self.SAMPLING_RATE)) / self.SAMPLING_RATE
        audio = (
            0.1 * np.sin(2 * np.pi * 220.0 * t)
            + 0.05 * np.sin(2 * np.pi * 440.0 * t)
            + 0.02 * np.sin(2 * np.pi * 880.0 * t)
        )
        return audio.astype(np.float32)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SeamlessM4T v2 text decoder.

        Runs the speech encoder on host to produce ``encoder_hidden_states`` and
        builds a single-step decoder input. The encoder is run in whatever dtype
        the loaded model uses, so the produced ``encoder_hidden_states`` already
        match the text decoder's dtype.

        Args:
            dtype_override: Optional torch.dtype. If the model is not loaded yet,
                            it is loaded with this dtype.
            batch_size: Batch size to replicate inputs to (default 1).

        Returns:
            dict: ``input_ids`` and ``encoder_hidden_states`` for the text decoder.
        """
        # Ensure processor + full model are available (e.g. standalone use).
        if self.processor is None or self.config is None:
            self._load_processor_and_config()
        if self.full_model is None:
            self.load_model(dtype_override=dtype_override)

        # Match the encoder run dtype to the loaded model's dtype.
        model_dtype = next(self.full_model.speech_encoder.parameters()).dtype

        audio = self._synthesize_audio()
        audio_inputs = self.processor(
            audio=audio,
            sampling_rate=self.SAMPLING_RATE,
            return_tensors="pt",
        )
        input_features = audio_inputs.input_features.to(model_dtype)
        attention_mask = audio_inputs.get("attention_mask")

        # Produce encoder hidden states on host.
        with torch.no_grad():
            encoder_outputs = self.full_model.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
            )
        encoder_hidden_states = encoder_outputs[0]

        # Single-step decoder input (BOS). Generation is out of scope for this
        # single-forward-pass bringup.
        bos_token_id = self.processor.tokenizer.bos_token_id
        decoder_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long)

        if batch_size > 1:
            decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                batch_size, dim=0
            )

        return {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def decode_output(self, outputs, inputs=None):
        """Decode the text decoder output into a first-step token prediction.

        The text decoder emits a hidden state; this projects it through the full
        model's ``lm_head`` and returns the predicted token string. Generation
        (autoregressive decoding) is out of scope for this bringup.

        Args:
            outputs: Output of the text decoder forward pass.
            inputs: Unused; accepted for interface compatibility.

        Returns:
            str: A human-readable summary of the first-step prediction.
        """
        if self.full_model is None:
            self.load_model()

        last_hidden_state = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        lm_head = self.full_model.lm_head
        last_hidden_state = last_hidden_state.to(
            next(lm_head.parameters()).dtype
        )
        logits = lm_head(last_hidden_state)
        predicted_id = int(logits[0, -1].float().argmax())
        token = self.processor.tokenizer.decode(
            [predicted_id], skip_special_tokens=False
        )
        return (
            f"SeamlessM4Tv2 first-step prediction: token_id={predicted_id} "
            f'token="{token}" | logits shape={tuple(logits.shape)}'
        )
