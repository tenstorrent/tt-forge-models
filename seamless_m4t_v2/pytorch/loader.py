# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SeamlessM4T v2 model loader implementation for speech-to-text translation.

SeamlessM4Tv2Model is a multi-component speech/text translation model. This
loader brings up the text decoder submodule (the autoregressive transformer
that produces text tokens), which is the key component on the speech-to-text
path: speech_encoder -> text_decoder. Encoder hidden states are produced on
host by running the speech encoder, exactly as the v1 SeamlessM4T loader does.
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

    LARGE = "large"


class ModelLoader(ForgeModel):
    """SeamlessM4T v2 model loader implementation for speech-to-text translation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/seamless-m4t-v2-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE

    # 16 kHz is the sampling rate SeamlessM4T expects.
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
        self._dtype = None

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
            self._dtype = dtype_override

        self.full_model.eval()

        # Return text_decoder submodule only
        return self.full_model.text_decoder

    def _generate_sample_audio(self):
        """Generate a deterministic synthetic 16 kHz mono waveform.

        torchaudio is intentionally not used here: in this environment its
        prebuilt extension is incompatible with the installed torch and fails
        to load, and pinning it risks moving the torch stack. A synthetic
        waveform is sufficient to exercise the speech-encoder front-end and
        produce encoder hidden states for the decoder.

        Returns:
            torch.Tensor: shape (1, num_samples) float32 waveform.
        """
        # ~2 seconds of audio: a couple of sine tones so the mel front-end has
        # non-trivial spectral content. Deterministic (no RNG) for reproducibility.
        num_samples = 2 * self.SAMPLING_RATE
        t = torch.arange(num_samples, dtype=torch.float32) / self.SAMPLING_RATE
        waveform = 0.5 * torch.sin(2 * torch.pi * 220.0 * t) + 0.3 * torch.sin(
            2 * torch.pi * 440.0 * t
        )
        return waveform.unsqueeze(0)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SeamlessM4T v2 text decoder.

        Args:
            dtype_override: Optional torch.dtype for the float-valued inputs
                            (encoder hidden states).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input arguments that can be fed to the text decoder.
        """
        # Ensure processor and full model are loaded (load_model must run first).
        if self.processor is None or self.full_model is None:
            raise RuntimeError(
                "Model and processor must be loaded before loading inputs"
            )

        # Build a synthetic audio waveform and run it through the processor.
        # (transformers >=5 renamed the `audios` processor kwarg to `audio`.)
        audio = self._generate_sample_audio()
        audio_inputs = self.processor(
            audio=audio.squeeze(0).numpy(),
            sampling_rate=self.SAMPLING_RATE,
            return_tensors="pt",
        )

        # The speech encoder runs in the model's dtype; feature dtype must match.
        model_dtype = self._dtype if self._dtype is not None else torch.float32
        input_features = audio_inputs.input_features.to(model_dtype)
        attention_mask = audio_inputs.get("attention_mask", None)

        # Run encoder to get encoder_hidden_states (host-side).
        with torch.no_grad():
            encoder_outputs = self.full_model.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
            )
        encoder_hidden_states = encoder_outputs[0]

        # Prepare decoder input IDs (start with BOS).
        tokenizer = self.processor.tokenizer
        bos_token_id = tokenizer.bos_token_id
        decoder_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long)

        # Add batch dimension if batch_size > 1.
        if batch_size > 1:
            decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                batch_size, dim=0
            )

        if dtype_override is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        # Arguments are inputs for the text decoder submodule.
        arguments = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }

        return arguments

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable text.

        The text decoder returns hidden states; project them through the model's
        lm_head to obtain token logits, then argmax-decode the first step.

        Args:
            outputs: Model output from a forward pass of the text decoder.

        Returns:
            str: Decoded output information
        """
        if self.processor is None:
            self._load_processor_and_config()

        # text_decoder returns last_hidden_state; project to vocab via lm_head.
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if self.full_model is not None:
            logits = self.full_model.lm_head(hidden)
        else:
            logits = hidden

        predicted_ids = torch.argmax(logits, dim=-1)

        decoded_text = self.processor.tokenizer.decode(
            predicted_ids[0], skip_special_tokens=True
        )

        return f"""
        SeamlessM4Tv2 Output:
          - Decoded text (first step): "{decoded_text}"
          - Output shape: {logits.shape}
          - Predicted token IDs: {predicted_ids[0].tolist()}
        """
