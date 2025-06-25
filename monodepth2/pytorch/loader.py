# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Monodepth2 model loader implementation
"""

from ...base import ForgeModel
from .src.utils import load_model, load_input


class ModelLoader(ForgeModel):
    """Loads Monodepth2 model and sample input."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "mono_640x192"
        self._height = None
        self._width = None

    def load_model(self, dtype_override=None):
        """Load pretrained Monodepth2 model."""
        model, height, width = load_model(self.model_name)
        model.eval()

        self._height = height
        self._width = width

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Monodepth2 model"""

        inputs = load_input(self._height, self._width)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
