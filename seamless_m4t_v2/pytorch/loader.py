# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SeamlessM4Tv2 model loader implementation for speech-to-text translation
"""
import io
import urllib.request
import wave
from typing import Optional

import numpy as np
import torch

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
    """Available SeamlessM4Tv2 model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """SeamlessM4Tv2 model loader implementation for speech-to-text translation tasks.

    SeamlessM4Tv2 is a multi-component speech/text translation pipeline (speech
    encoder, text encoder, text decoder, text-to-unit model and HiFi-GAN vocoder).
    This loader brings up the ``text_decoder`` submodule as a single forward pass:
    a speech clip is encoded on CPU to produce ``encoder_hidden_states`` which, with
    the decoder start token, are fed to the text decoder.
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
        """Load and return the SeamlessM4Tv2 text decoder submodule for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The SeamlessM4Tv2 text decoder submodule.
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

    def get_mesh_config(self, num_devices: int):
        """Return the (mesh_shape, mesh_axis_names) for tensor-parallel sharding.

        The text decoder is sharded along the model axis with a 1-D model mesh
        (``(1, num_devices)``). The number of attention heads must be divisible by
        the model-axis size for the Megatron column/row split to be valid.

        Args:
            num_devices: Number of devices to distribute the model across.

        Returns:
            tuple: (mesh_shape, mesh_axis_names)
        """
        if self.config is None:
            self._load_processor_and_config()

        mesh_shape = (1, num_devices)
        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron column->row tensor-parallel shard spec for the text decoder.

        ``model`` is the text decoder returned by ``load_model``. Self-attention and
        cross-attention q/k/v projections are column-sharded on the model axis and
        their output projections row-sharded; the FFN up-projection (fc1) is column
        sharded and the down-projection (fc2) row sharded.

        Args:
            model: The text decoder module (on device).

        Returns:
            Dict[Tensor, Tuple[str, str]]: weight -> partition spec.
        """
        shard_specs = {}
        for layer in model.layers:
            # Self-attention
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.out_proj.weight] = ("batch", "model")

            # Cross-attention (attends to the speech encoder hidden states)
            shard_specs[layer.cross_attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.cross_attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.cross_attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.cross_attention.out_proj.weight] = ("batch", "model")

            # Feed-forward
            shard_specs[layer.ffn.fc1.weight] = ("model", "batch")
            shard_specs[layer.ffn.fc2.weight] = ("batch", "model")
        return shard_specs

    @staticmethod
    def _decode_wav(audio_bytes):
        """Decode PCM WAV bytes into a float waveform tensor without torchcodec/FFmpeg.

        Args:
            audio_bytes: Raw bytes of a PCM WAV file.

        Returns:
            tuple: (waveform tensor of shape [channels, num_samples], sample_rate)
        """
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            num_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        # Map sample width (bytes) to the corresponding signed-int numpy dtype
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        # Normalize to [-1, 1] based on the integer full-scale range
        data /= float(np.iinfo(dtype).max)

        # De-interleave channels -> shape [channels, num_samples]
        data = data.reshape(-1, num_channels).T
        return torch.from_numpy(data.copy()), sample_rate

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SeamlessM4Tv2 text decoder.

        Args:
            dtype_override: Optional torch.dtype to cast the float inputs
                            (encoder hidden states) to match the model dtype.
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

        # Load and format audio input. The clip is a standard PCM WAV, decoded with
        # the stdlib `wave` module so we don't depend on torchaudio's torchcodec/
        # FFmpeg backend (which may be missing system libs in CI).
        url = "https://courses.cs.duke.edu/cps001/spring06/class/06_Sound/sounds/preamble.wav"
        with urllib.request.urlopen(url) as response:
            audio_data = response.read()
        audio, orig_freq = self._decode_wav(audio_data)

        # Resample audio to the model's expected 16 kHz sampling rate
        audio = torchaudio.functional.resample(
            audio, orig_freq=orig_freq, new_freq=16_000
        )

        # Process audio into input features
        audio_inputs = self.processor(
            audio=audio, sampling_rate=16_000, return_tensors="pt"
        )

        # Run the speech encoder on CPU to get encoder_hidden_states
        with torch.no_grad():
            encoder_outputs = self.full_model.speech_encoder(
                input_features=audio_inputs.input_features,
                attention_mask=audio_inputs.attention_mask,
            )
        encoder_hidden_states = encoder_outputs[0]

        # Prepare decoder input IDs (decoder starts from BOS)
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

        The text decoder returns hidden states (``last_hidden_state``), not vocab
        logits. To produce a meaningful first-step token prediction we project the
        decoder hidden states through the full model's ``lm_head``.

        Args:
            outputs: Model output from a text-decoder forward pass

        Returns:
            str: Decoded output information
        """
        if self.processor is None:
            self._load_processor_and_config()

        # Pull the decoder hidden states out of the output structure
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

        # Project to vocab logits via the model's LM head for a real token prediction
        with torch.no_grad():
            lm_head = self.full_model.lm_head.to(hidden_states.dtype)
            logits = lm_head(hidden_states)

        # Get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode to text
        decoded_text = self.processor.tokenizer.decode(
            predicted_ids[0], skip_special_tokens=True
        )

        return f"""
        SeamlessM4Tv2 Output:
          - Decoded text: "{decoded_text}"
          - Output shape: {logits.shape}
          - Predicted token IDs: {predicted_ids[0].tolist()}
        """
