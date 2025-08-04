# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBERT model loader implementation for masked language modeling.
"""

from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available DistilBERT model variants for masked language modeling."""

    DISTILBERT_BASE_CASED = "distilbert-base-cased"
    DISTILBERT_BASE_UNCASED = "distilbert-base-uncased"
    DISTILBERT_BASE_MULTILINGUAL_CASED = "distilbert-base-multilingual-cased"


class ModelLoader(ForgeModel):
    """DistilBERT model loader implementation for masked language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DISTILBERT_BASE_CASED: LLMModelConfig(
            pretrained_model_name="distilbert-base-cased",
            max_length=128,
        ),
        ModelVariant.DISTILBERT_BASE_UNCASED: LLMModelConfig(
            pretrained_model_name="distilbert-base-uncased",
            max_length=128,
        ),
        ModelVariant.DISTILBERT_BASE_MULTILINGUAL_CASED: LLMModelConfig(
            pretrained_model_name="distilbert-base-multilingual-cased",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DISTILBERT_BASE_CASED

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.input_prompt = "The capital of France is [MASK]."
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DistilBERT-MaskedLM",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load DistilBERT model for masked language modeling from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DistilBERT model instance.
        """

        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DistilBertForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for DistilBERT masked language modeling.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            # Ensure tokenizer is initialized
            self.load_model(dtype_override=dtype_override)

        # Data preprocessing
        inputs = self.tokenizer(
            self.input_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for masked language modeling."""
        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        print(
            "The predicted token for the [MASK] is: ",
            self.tokenizer.decode(predicted_token_id),
        )
