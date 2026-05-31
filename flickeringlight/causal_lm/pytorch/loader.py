# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FlickeringLight-14B model loader implementation for causal language modeling.

The published artifact (mradermacher/FlickeringLight-14B-i1-GGUF) is an imatrix
GGUF quantization of yamatazen/FlickeringLight-14B, a Mistral-Nemo based
frankenmerge. The weights are loaded straight from a GGUF file via the
transformers GGUF integration, which dequantizes the quantized blocks back into
regular torch tensors so the model can run on CPU and on the Tenstorrent device.
"""
import contextlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available FlickeringLight model variants (by GGUF quantization)."""

    I1_Q4_K_M = "i1_q4_k_m"


class ModelLoader(ForgeModel):
    """FlickeringLight-14B GGUF loader for causal language modeling tasks."""

    # GGUF file that backs each variant inside the GGUF repository.
    _GGUF_FILES = {
        ModelVariant.I1_Q4_K_M: "FlickeringLight-14B.i1-Q4_K_M.gguf",
    }

    # Dictionary of available model variants. The repo holds only GGUF files; the
    # config/tokenizer are read from the GGUF metadata itself.
    _VARIANTS = {
        ModelVariant.I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/FlickeringLight-14B-i1-GGUF",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.I1_Q4_K_M

    @staticmethod
    @contextlib.contextmanager
    def _low_memory_gguf_dequant():
        """Halve the peak host memory of GGUF loading.

        transformers dequantizes every GGUF tensor to fp32 and keeps the full
        intermediate state dict in RAM while the (bf16) model is being built on
        top of it. For a 14B model the fp32 dict alone is ~58GB, which OOMs the
        runner host before the device test can even start. Emitting the
        dequantized weights as fp16 instead halves that intermediate dict
        (~29GB) while leaving transformers' architecture-specific tensor
        reshaping/mapping untouched. The model is still materialized in the
        requested dtype (bf16) for the actual run.
        """
        import gguf

        original_dequantize = gguf.dequantize

        def _dequantize_to_fp16(data, qtype):
            return original_dequantize(data, qtype).astype(np.float16)

        gguf.dequantize = _dequantize_to_fp16
        try:
            yield
        finally:
            gguf.dequantize = original_dequantize

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="FlickeringLight-14B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF metadata.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the FlickeringLight model instance for this variant.

        The model weights are dequantized from the GGUF file by the transformers
        GGUF integration.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.

        Returns:
            torch.nn.Module: The model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded (also primes the GGUF download/cache).
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        # Default to bfloat16 to keep the host memory footprint of this 14B
        # model manageable and to match the dtype used on device.
        model_kwargs["dtype"] = dtype_override if dtype_override is not None else torch.bfloat16
        model_kwargs |= kwargs

        with self._low_memory_gguf_dequant():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FlickeringLight model.

        Args:
            dtype_override: Optional torch.dtype (unused for integer token inputs,
                            kept for interface compatibility).
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Short prompt keeps the sequence small for first bringup on device.
        test_input = "The future of artificial intelligence is"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass.

        Returns:
            str: Decoded next token text.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
