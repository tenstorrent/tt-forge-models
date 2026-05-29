# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AllKingsRoleplay model loader implementation for causal language modeling.

AllKingsRoleplay_12b_v1 is a Mistral-Nemo (12B) class mergekit merge. This loader
brings up the GGUF (imatrix) quantized release published by ``mradermacher``. The
model weights are loaded from the GGUF file via transformers' ``gguf_file`` path
(which dequantizes to a regular PyTorch ``MistralForCausalLM``), while the tokenizer
is loaded from the unquantized base repo which ships a ``tokenizer.json``.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available AllKingsRoleplay model variants for causal LM."""

    # i1 (imatrix) GGUF quant of AllKingsRoleplay_12b_v1.
    V1_12B_I1_Q4_K_M = "12b_v1_i1_Q4_K_M"


class ModelLoader(ForgeModel):
    """AllKingsRoleplay model loader for causal language modeling tasks."""

    # GGUF repo that hosts the quantized weights for each variant.
    _GGUF_REPO = "mradermacher/AllKingsRoleplay_12b_v1-i1-GGUF"

    # Specific GGUF file to load per variant.
    _GGUF_FILE = {
        ModelVariant.V1_12B_I1_Q4_K_M: "AllKingsRoleplay_12b_v1.i1-Q4_K_M.gguf",
    }

    # Unquantized base repo, used only for the tokenizer (ships tokenizer.json).
    _TOKENIZER_REPO = "kainatq/AllKingsRoleplay_12b_v1"

    # Dictionary of available model variants using structured configs.
    _VARIANTS = {
        ModelVariant.V1_12B_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/AllKingsRoleplay_12b_v1-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V1_12B_I1_Q4_K_M

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to keep. If None, uses the
                     model's full layer count.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="AllKingsRoleplay",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._TOKENIZER_REPO, **tokenizer_kwargs
        )

        # Ensure a pad token exists for batched/padded inputs.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, num_layers=None, **kwargs):
        """Load and return the AllKingsRoleplay model instance for this variant.

        Args:
            dtype_override: Optional dtype to cast the model to. If not provided,
                     the dequantized GGUF weights default to float32.
            num_layers: Optional number of hidden layers to keep. If None, uses the
                     instance value (which may also be None for the full model).

        Returns:
            torch.nn.Module: The AllKingsRoleplay model instance for causal LM.
        """
        gguf_file = self._GGUF_FILE[self._variant]

        # Ensure tokenizer is loaded.
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(self._GGUF_REPO, **model_kwargs)

        # Optionally truncate the layer stack (useful for bringup smoke tests).
        layers_to_keep = num_layers if num_layers is not None else self.num_layers
        if layers_to_keep is not None:
            model.model.layers = model.model.layers[:layers_to_keep]
            model.config.num_hidden_layers = layers_to_keep

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AllKingsRoleplay model.

        Args:
            dtype_override: Optional dtype to cast floating-point inputs to.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized.
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size.
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested (applies to float tensors only).
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def load_config(self):
        """Load and return the configuration for the AllKingsRoleplay model variant.

        The config is read from the unquantized base repo (the GGUF repo has no
        ``config.json``), and is consistent with the GGUF metadata.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(self._TOKENIZER_REPO)
        return self.config
