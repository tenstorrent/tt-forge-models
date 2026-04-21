# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PE-AV (Perception Encoder Audio-Video) model loader implementation for joint
audio-visual-text embedding.

Reference: https://huggingface.co/facebook/pe-av-large
"""
import os
import tempfile
from typing import Optional

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
    """Available PE-AV model variants."""

    PE_AV_LARGE = "pe-av-large"


class ModelLoader(ForgeModel):
    """PE-AV model loader for joint audio-visual-text embedding tasks."""

    _VARIANTS = {
        ModelVariant.PE_AV_LARGE: ModelConfig(
            pretrained_model_name="facebook/pe-av-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PE_AV_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PE_AV",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_AUDIO_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import PeAudioVideoProcessor

        self.processor = PeAudioVideoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PE-AV model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The PE-AV model instance.
        """
        from transformers import PeAudioVideoModel

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = PeAudioVideoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    @staticmethod
    def _create_synthetic_audio(sampling_rate=16000, duration=1.0):
        """Create a temporary synthetic wav file and return its path."""
        import numpy as np
        import soundfile as sf

        num_samples = int(sampling_rate * duration)
        audio = np.random.randn(num_samples).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, audio, sampling_rate)
        return tmp.name

    @staticmethod
    def _create_synthetic_video(num_frames=16, height=224, width=224, fps=8):
        """Create a temporary synthetic mp4 video file and return its path."""
        import cv2
        import numpy as np

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return tmp.name

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the PE-AV model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors for audio, video, and text modalities.
        """
        if self.processor is None:
            self._load_processor()

        video_path = self._create_synthetic_video()
        audio_path = self._create_synthetic_audio()

        self.text_prompts = ["a dog barking in the park"]

        try:
            inputs = self.processor(
                videos=[video_path],
                text=self.text_prompts,
                audio=[audio_path],
                return_tensors="pt",
                padding=True,
            )
        finally:
            os.unlink(video_path)
            os.unlink(audio_path)

        inputs = dict(inputs)

        if batch_size > 1:
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process PE-AV outputs to report audio-text similarity scores.

        Args:
            outputs: Raw model output from PeAudioVideoModel.
        """
        if self.text_prompts is None:
            self.text_prompts = ["a dog barking in the park"]

        logits = None
        if isinstance(outputs, tuple):
            logits = next(
                (t for t in outputs if torch.is_tensor(t) and t.ndim == 2),
                None,
            )
        elif hasattr(outputs, "logits_audio_text"):
            logits = outputs.logits_audio_text

        if logits is None:
            return

        probs = logits.sigmoid()
        for i, text in enumerate(self.text_prompts):
            if i < probs.shape[-1]:
                print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass.

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass.
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
                elif hasattr(item, "last_hidden_state"):
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
