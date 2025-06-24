# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DPR model loader implementation
"""

import torch
from transformers import DPRReader, DPRReaderTokenizer
from ...base import ForgeModel


class ModelLoader(ForgeModel):

    # Shared configuration parameters
    model_name = "facebook/dpr-reader-single-nq-base"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the DPR Reader model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DPR Reader model instance.

        """

        # Initialize tokenizer
        cls.tokenizer = DPRReaderTokenizer.from_pretrained(cls.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DPRReader.from_pretrained(
            cls.model_name, return_dict=False, **model_kwargs
        )
        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DPR Reader model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model(dtype_override=dtype_override)

        inputs = cls.tokenizer(
            questions=["What is love ?"],
            titles=["Haddaway"],
            texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
