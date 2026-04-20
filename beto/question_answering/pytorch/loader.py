# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BETO model loader implementation for question answering (Spanish SQuAD2).
"""

from transformers import BertForQuestionAnswering, BertTokenizer
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
    """Available BETO model variants for question answering."""

    MRM8488_BERT_BASE_SPANISH_WWM_CASED_FINETUNED_SPA_SQUAD2_ES = (
        "mrm8488_bert_base_spanish_wwm_cased_finetuned_spa_squad2_es"
    )


class ModelLoader(ForgeModel):
    """BETO model loader implementation for Spanish question answering."""

    _VARIANTS = {
        ModelVariant.MRM8488_BERT_BASE_SPANISH_WWM_CASED_FINETUNED_SPA_SQUAD2_ES: LLMModelConfig(
            pretrained_model_name="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.MRM8488_BERT_BASE_SPANISH_WWM_CASED_FINETUNED_SPA_SQUAD2_ES
    )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        self.context = (
            "Manuel Romero ha colaborado activamente con el repositorio de "
            "transformers de HuggingFace. BETO es un modelo BERT entrenado "
            "sobre un gran corpus en español por el equipo de la Universidad "
            "de Chile y afinado en SQuAD-es-v2.0 para la tarea de preguntas "
            "y respuestas."
        )
        self.question = "¿Para qué tarea se afinó BETO en este modelo?"

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
            model="BETO",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BETO model for question answering from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BETO model instance for question answering.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for BETO question answering.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.question,
            self.context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for question answering."""
        inputs = self.load_inputs()
        start_logits = co_out[0]
        end_logits = co_out[1]

        answer_start_index = start_logits.argmax()
        answer_end_index = end_logits.argmax()

        input_ids = inputs["input_ids"]
        predict_answer_tokens = input_ids[0, answer_start_index : answer_end_index + 1]

        predicted_answer = self.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )
        print("Predicted answer:", predicted_answer)
