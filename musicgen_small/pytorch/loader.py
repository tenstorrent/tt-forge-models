# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Musicgen-small model loader implementation
"""
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """Musicgen-small model loader implementation."""

    # Shared configuration parameters
    model_name = "facebook/musicgen-small"

    @classmethod
    def load_model(cls):
        """Load and return the Musicgen-small model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Musicgen-small model instance.

        """
        cls.processor = AutoProcessor.from_pretrained(cls.model_name)
        cls.model = MusicgenForConditionalGeneration.from_pretrained(cls.model_name)
        return cls.model

    @classmethod
    def load_inputs(cls, batch_size=1):
        """Load and return sample inputs for the Musicgen-small model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if not hasattr(cls, "processor"):
            cls.load_model()

        inputs = cls.processor(
            text=[
                "80s pop track with bassy drums and synth",
                "90s rock song with loud guitars and heavy drums",
            ],
            padding=True,
            return_tensors="pt",
        )

        # If batch_size is different from 2, adjust using repeat_interleave
        if batch_size != 2:
            # Calculate how many times to repeat each example
            repeats_per_example = batch_size // 2
            remaining = batch_size % 2

            # Apply repeat_interleave to input tensors
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    if remaining == 0:
                        # Even division
                        inputs[key] = inputs[key].repeat_interleave(
                            repeats_per_example, dim=0
                        )
                    else:
                        # Handle remainder by repeating first example one extra time
                        repeated = inputs[key].repeat_interleave(
                            repeats_per_example, dim=0
                        )
                        extra = inputs[key][:1].repeat_interleave(remaining, dim=0)
                        inputs[key] = torch.cat([repeated, extra], dim=0)

        pad_token_id = cls.model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones(
                (
                    inputs.input_ids.shape[0] * cls.model.decoder.num_codebooks,
                    1,
                ),
                dtype=torch.long,
            )
            * pad_token_id
        )

        inputs["max_new_tokens"] = 1
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
