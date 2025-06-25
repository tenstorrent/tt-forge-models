# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Pixtral model loader implementation
"""


import torch
from transformers import LlavaForConditionalGeneration  # , AutoProcessor
from ....base import ForgeModel


class ModelLoader(ForgeModel):

    # Shared configuration parameters
    model_name = "mistral-community/pixtral-12b"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Mistral Pixtral model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Mistral Pixtral model instance.

        """
        # self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = LlavaForConditionalGeneration.from_pretrained(
            cls.model_name, **model_kwargs
        )
        return model

    @classmethod
    def load_inputs(cls):
        """Load and return sample inputs for the Mistral Pixtral model with default settings.

        Returns:
            dict: Input tensors that can be fed to the model.
        """

        # https://github.com/tenstorrent/tt-torch/issues/904
        inputs = {
            "input_ids": torch.tensor(
                [[1, 3, 12483, 1593, 11386, 10, 51883, 3226, 1063, 10, 4]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long
            ),
        }

        return inputs
