# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 model loader implementation for the (text-only) causal language modeling path.

Molmo2 (allenai/Molmo2-8B) is a multimodal image-text-to-text model whose text
decoder is a Qwen3-based decoder-only transformer. This loader exercises the text
decoder (+ LM head) of ``Molmo2ForConditionalGeneration`` by running a text-only
forward pass (``pixel_values=None``), which is the compute-dominant component of
the model. The vision tower is brought up separately under ``molmo2/vision``.

The model ships as custom HuggingFace code (``trust_remote_code=True``); the
revision is pinned for reproducibility.
"""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoConfig

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
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Molmo2 variants for the causal LM (text) path."""

    MOLMO2_8B = "8b"


class ModelLoader(ForgeModel):
    """Molmo2 text-decoder loader for causal language modeling tasks."""

    # Pinned revision of the custom-code repo for reproducibility.
    _REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # Sample text for the text-only causal LM forward pass.
    sample_text = "The capital of France is"

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
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer_kwargs = {"trust_remote_code": True, "revision": self._REVISION}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Molmo2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                dtype. If not provided, the model uses its checkpoint dtype.

        Returns:
            torch.nn.Module: The Molmo2ForConditionalGeneration model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True, "revision": self._REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        # Disable KV-cache for a clean single forward-pass graph on the static device path.
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return text-only sample inputs for the Molmo2 causal LM path.

        Args:
            dtype_override: Optional torch.dtype for input casting (applies to
                floating-point tensors only).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")
        # Keep only the text inputs the decoder needs (no pixel_values).
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Pad to a fixed length so the device graph has static shapes.
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len
        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def unpack_forward_output(self, output):
        """Unwrap Molmo2CausalLMOutputWithPast to the logits tensor for comparison."""
        if isinstance(output, torch.Tensor):
            return output
        return output.logits

    def load_config(self):
        """Load and return the configuration for the Molmo2 variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=self._REVISION,
        )
        return self.config
