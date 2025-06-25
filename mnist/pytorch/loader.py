# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MNIST model loader implementation
"""

from ...base import ForgeModel
from .src.utils import load_model, load_input


class ModelLoader(ForgeModel):
    """Loads MNIST model and sample input."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    def load_model(self, dtype_override=None):
        """Load pretrained MNIST model."""
        model = load_model()
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for MNIST model"""

        inputs = load_input()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
