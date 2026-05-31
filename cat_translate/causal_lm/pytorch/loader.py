# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CAT-Translate model loader implementation for causal language modeling.

CAT-Translate-7b (cyberagent/CAT-Translate-7b) is a Mistral-architecture
(MistralForCausalLM) Japanese<->English machine-translation model. This loader
brings up the imatrix-quantized GGUF redistribution
(mradermacher/CAT-Translate-7b-i1-GGUF): transformers loads and de-quantizes the
GGUF weights into a full-precision MistralForCausalLM at load time.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    """Available CAT-Translate model variants."""

    I1_Q4_K_M = "7b_i1_q4_k_m"


class ModelLoader(ForgeModel):
    """CAT-Translate model loader for causal language modeling (translation) tasks."""

    # GGUF redistribution repo and the specific quantized file per variant.
    # transformers de-quantizes the GGUF into a full MistralForCausalLM on load.
    _GGUF_REPO = "mradermacher/CAT-Translate-7b-i1-GGUF"
    _GGUF_FILES = {
        ModelVariant.I1_Q4_K_M: "CAT-Translate-7b.i1-Q4_K_M.gguf",
    }

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/CAT-Translate-7b-i1-GGUF",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.I1_Q4_K_M

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
            model="CAT-Translate",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """Return the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF checkpoint.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
            padding_side="right",
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the CAT-Translate model instance for this instance's variant.

        The model is loaded from a GGUF file; transformers de-quantizes the
        weights into a full-precision MistralForCausalLM.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.
                            If not provided, the model uses bfloat16.

        Returns:
            torch.nn.Module: The CAT-Translate model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CAT-Translate model.

        Args:
            dtype_override: Optional torch.dtype (unused for integer token inputs,
                            accepted for interface consistency).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Translation-style prompt for the Japanese<->English translation model.
        test_input = "Translate the following text to Japanese: How are you today?"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded next token text
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
