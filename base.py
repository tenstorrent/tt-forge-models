# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Base class for all model implementations in tt-forge-models
"""
from abc import ABC, abstractmethod
import torch


class ForgeModel(ABC):
    """Base class for all model implementations that can be shared across Tenstorrent projects."""

    # Models can override these class variables to support variants
    _VARIANTS = {}  # Empty by default for models without variants
    DEFAULT_VARIANT = None

    def __init__(self, variant=None):
        """Initialize the model loader with an optional variant.

        Args:
            variant: Optional string specifying which variant to use for this instance.
                    If None, the class's DEFAULT_VARIANT will be used.
        """
        self._variant = self._validate_variant(variant)
        self._variant_config = self.get_variant_config(self._variant)

    @classmethod
    def query_available_variants(cls):
        """Returns a dictionary of available model variants and their descriptions.

        Returns:
            dict: Dictionary mapping variant names to descriptions, or empty dict if
                  the model doesn't support variants.
        """
        if not cls._VARIANTS:
            return {}
        return {
            variant: config.get("description", "")
            for variant, config in cls._VARIANTS.items()
        }

    @classmethod
    def _validate_variant(cls, variant=None):
        """Validates and returns the variant to use.

        Args:
            variant: Optional string specifying which variant to validate.

        Returns:
            str or None: Validated variant name, or None for models without variants.

        Raises:
            ValueError: If the specified variant doesn't exist.
        """
        # If model doesn't support variants, return None
        if not cls._VARIANTS:
            return None

        # Use default if none specified
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Validate the variant exists
        if variant not in cls._VARIANTS:
            valid_variants = list(cls._VARIANTS.keys())
            raise ValueError(
                f"Invalid variant '{variant}'. Available variants: {valid_variants}"
            )

        return variant

    @classmethod
    def get_variant_config(cls, variant=None):
        """Get configuration for a specific variant after validation.

        Args:
            variant: Optional string specifying which variant to get config for.

        Returns:
            dict or None: Variant configuration dictionary, or None for models without variants.
        """
        variant = cls._validate_variant(variant)
        if variant is None:
            return None

        return cls._VARIANTS[variant]

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and return the model instance using this instance's variant.

        Args:
            **kwargs: Additional model-specific arguments.

        Returns:
            torch.nn.Module: The model instance
        """
        pass

    @abstractmethod
    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the model using this instance's variant.

        Args:
            **kwargs: Additional input-specific arguments.

        Returns:
            Any: Sample inputs that can be fed to the model
        """
        pass

    @classmethod
    @abstractmethod
    def decode_output(cls, **kwargs):
        """Load and return sample inputs for the model.

        Returns:
            Any: Output will be Decoded from the model
        """
        pass
