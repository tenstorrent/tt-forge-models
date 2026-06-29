# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SeamlessM4T v2 model loader implementation for speech-to-text translation
"""
import io
import urllib.request
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SeamlessM4T v2 model variants."""

    LARGE = "large"


class ModelLoader(ForgeModel):
    """SeamlessM4T v2 model loader implementation for speech-to-text translation tasks.

    SeamlessM4Tv2Model is a multi-component model (text encoder, speech encoder,
    text decoder, text-to-unit model and a HiFi-GAN vocoder). The headline bringup
    target is the ``text_decoder`` submodule (a single transformer-decoder forward
    pass), mirroring the SeamlessM4T v1 loader.
    """

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/seamless-m4t-v2-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE

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
        """Load and return the SeamlessM4T v2 text decoder submodule for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The SeamlessM4T v2 text decoder submodule.
        """
        from transformers import SeamlessM4Tv2Model

        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Ensure processor and config are loaded
        if self.processor is None or self.config is None:
            self._load_processor_and_config()

        # Load full model
        self.full_model = SeamlessM4Tv2Model.from_pretrained(
            model_name, config=self.config, **kwargs
        )

        if dtype_override is not None:
            self.full_model = self.full_model.to(dtype_override)

        # Return text_decoder submodule only
        return self.full_model.text_decoder

    @staticmethod
    def _load_wav(audio_bytes):
        """Decode a PCM WAV byte stream into a (channels, samples) float tensor.

        Uses the stdlib ``wave`` module so we don't depend on FFmpeg/torchcodec
        (torchaudio>=2.10 routes ``torchaudio.load`` through torchcodec, which
        needs system FFmpeg shared libs that aren't guaranteed to be present).
        """
        import wave

        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            orig_freq = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        dtype_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
        if sampwidth not in dtype_map:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth}")
        data = torch.frombuffer(bytearray(frames), dtype=dtype_map[sampwidth]).float()
        if sampwidth == 2:
            data = data / 32768.0
        elif sampwidth == 4:
            data = data / 2147483648.0
        else:  # 8-bit PCM is unsigned, centered at 128
            data = (data - 128.0) / 128.0
        # De-interleave channels -> (channels, samples)
        audio = data.view(-1, n_channels).t().contiguous()
        return audio, orig_freq

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SeamlessM4T v2 text decoder.

        Runs the speech encoder on a short audio clip to produce the cross-attention
        ``encoder_hidden_states`` and pairs it with a BOS decoder token.

        Args:
            dtype_override: Optional torch.dtype to cast the floating-point inputs.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input arguments that can be fed to the text decoder.
        """
        import torchaudio

        # Ensure processor and full model are loaded
        if self.processor is None or self.full_model is None:
            raise RuntimeError(
                "Model and processor must be loaded before loading inputs"
            )

        # Load and format audio input
        url = "https://courses.cs.duke.edu/cps001/spring06/class/06_Sound/sounds/preamble.wav"
        with urllib.request.urlopen(url) as response:
            audio_data = response.read()
        audio, orig_freq = self._load_wav(audio_data)

        # Resample audio to the model's 16 kHz sampling rate
        audio = torchaudio.functional.resample(
            audio, orig_freq=orig_freq, new_freq=16_000
        )

        # Process audio (transformers>=5 renamed the `audios` kwarg to `audio`)
        audio_inputs = self.processor(
            audio=audio, sampling_rate=16_000, return_tensors="pt"
        )
        input_features = audio_inputs.input_features
        if dtype_override is not None:
            input_features = input_features.to(dtype_override)

        # Run encoder to get encoder_hidden_states
        with torch.no_grad():
            encoder_outputs = self.full_model.speech_encoder(
                input_features=input_features,
                attention_mask=audio_inputs.get("attention_mask"),
            )
        encoder_hidden_states = encoder_outputs[0]

        # Prepare decoder input IDs (BOS token)
        tokenizer = self.processor.tokenizer
        bos_token_id = tokenizer.bos_token_id
        decoder_input_ids = torch.tensor([[bos_token_id]])

        # Add batch dimension if batch_size > 1
        if batch_size > 1:
            decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                batch_size, dim=0
            )

        if dtype_override is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        # Arguments are inputs for the text decoder submodule
        arguments = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }

        return arguments

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded output information
        """
        if self.processor is None:
            self._load_processor_and_config()

        # text_decoder returns last_hidden_state; project through the lm_head if
        # the full model is available, otherwise report the hidden-state shape.
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        if self.full_model is not None:
            logits = self.full_model.lm_head(hidden)
            predicted_ids = torch.argmax(logits, dim=-1)
            decoded_text = self.processor.tokenizer.decode(
                predicted_ids[0], skip_special_tokens=True
            )
            return f"""
        SeamlessM4Tv2 Output:
          - Decoded text: "{decoded_text}"
          - Logits shape: {logits.shape}
          - Predicted token IDs: {predicted_ids[0].tolist()}
        """

        return f"""
        SeamlessM4Tv2 Output:
          - Hidden state shape: {hidden.shape}
        """
