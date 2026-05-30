# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bielik model loader implementation for causal language modeling.

Bielik-4.5B-v3.0-Instruct is a Polish instruction-tuned LLM with a Llama
architecture. This loader consumes the GGUF distribution of the model; the
base (safetensors) repo is gated, while the GGUF repo is openly accessible.
Transformers dequantizes the GGUF weights and reconstructs both the config and
the tokenizer directly from the GGUF file metadata, so no access to the gated
base repo is required.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import torch

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Bielik model variants for causal LM."""

    V3_0_INSTRUCT_FP16 = "4.5B_v3.0_Instruct_fp16"


# GGUF file within the HuggingFace repo for each variant.
_GGUF_FILES = {
    ModelVariant.V3_0_INSTRUCT_FP16: "Bielik-4.5B-v3.0-Instruct-fp16.gguf",
}


class ModelLoader(ForgeModel):
    """Bielik model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.V3_0_INSTRUCT_FP16: LLMModelConfig(
            pretrained_model_name="speakleash/Bielik-4.5B-v3.0-Instruct-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V3_0_INSTRUCT_FP16

    # Sample text for causal LM (Polish; long enough to exercise the model with
    # a non-trivial number of real tokens, which keeps PCC stable).
    sample_text = (
        "Sztuczna inteligencja zmienia sposob, w jaki pracujemy i uczymy sie "
        "na co dzien. Napisz krotki akapit wyjasniajacy, dlaczego rozwoj "
        "dużych modeli jezykowych jest wazny dla rozwoju nauki i gospodarki w "
        "Polsce oraz jakie korzysci moga z tego plynac dla zwyklych ludzi."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None
        self.model = None

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
            model="Bielik",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = _GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Ensure a pad token exists for batched/padded inputs.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Bielik model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its native dtype.

        Returns:
            torch.nn.Module: The Bielik model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = _GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Bielik model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' float dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # For causal LM, we need both input_ids and attention_mask
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested (float tensors only)
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Use the unpadded real-token prompt. A long zero-padded region hurts
        # PCC for this deep (60-layer) model on device, so we keep the input
        # to its real tokens (no padding) which keeps PCC stable.
        self.seq_len = inputs["input_ids"].shape[1]
        return inputs

    def load_config(self):
        """Load and return the configuration for the Bielik model variant.

        The config is reconstructed from the GGUF file metadata.

        Returns:
            The configuration object for the Bielik model.
        """
        if self.model is not None:
            self.config = self.model.config
        return self.config
