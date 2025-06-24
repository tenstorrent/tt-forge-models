# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Base class for model loaders.

This module provides the ForgeModel base class with common functionality
for loading models, inputs, etc.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Type

from .config import ModelConfig, ModelInfo
import torch


class ForgeModel(ABC):
    """Base class for all TT-Forge model loaders."""

    # This is intended to be overridden by subclasses to define available model variants
    # Format: {"variant_name": ModelConfig(...), ...}
    _VARIANTS: Dict[
        str, ModelConfig
    ] = {}  # Empty by default for models without variants
    DEFAULT_VARIANT = None

    def __init__(self, variant=None):
        """Initialize a ForgeModel instance.

        Args:
            variant: Optional string specifying which variant to use.
                    If None, the default variant will be used.
        """
        # Validate and store the variant for this instance
        self._variant = self._validate_variant(variant)

        # Cache the variant configuration for efficiency
        self._variant_config = self.get_variant_config(variant)

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
            variant: config.description for variant, config in cls._VARIANTS.items()
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
    def get_variant_config(cls, variant=None) -> Optional[ModelConfig]:
        """Get configuration for a specific variant after validation.

        Args:
            variant: Optional string specifying which variant to get config for.

        Returns:
            ModelConfig or None: Variant configuration object, or None for models without variants.
        """
        variant = cls._validate_variant(variant)
        if variant is None:
            return None

        return cls._VARIANTS[variant]

    @classmethod
    def get_model_info(cls, variant=None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional string specifying which variant to get info for.

        Returns:
            ModelInfo: Information about the model and variant
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_model_info")

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
