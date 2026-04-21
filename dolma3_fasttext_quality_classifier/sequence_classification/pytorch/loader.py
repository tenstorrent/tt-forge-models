# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""allenai/dolma3-fasttext-quality-classifier model loader implementation for document quality classification."""

from typing import Optional

import fasttext
import torch
from huggingface_hub import hf_hub_download

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available dolma3-fasttext-quality-classifier model variants."""

    DOLMA3_FASTTEXT_QUALITY_CLASSIFIER = "dolma3-fasttext-quality-classifier"


class FastTextClassifierModule(torch.nn.Module):
    """PyTorch module wrapping the fastText quality classifier output layer.

    The fastText model is not a PyTorch module, so this wrapper reproduces the
    classifier head (a linear projection over the sentence vector followed by a
    softmax) using weights extracted from the loaded fastText model.
    """

    def __init__(self, output_matrix: torch.Tensor):
        super().__init__()
        num_labels, embedding_dim = output_matrix.shape
        self.classifier = torch.nn.Linear(embedding_dim, num_labels, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(output_matrix)

    def forward(self, sentence_vector: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(sentence_vector)
        return torch.softmax(logits, dim=-1)


class ModelLoader(ForgeModel):
    """allenai/dolma3-fasttext-quality-classifier loader for document quality classification."""

    _VARIANTS = {
        ModelVariant.DOLMA3_FASTTEXT_QUALITY_CLASSIFIER: ModelConfig(
            pretrained_model_name="allenai/dolma3-fasttext-quality-classifier",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOLMA3_FASTTEXT_QUALITY_CLASSIFIER

    sample_text = (
        "This well-structured article provides clear and accurate information "
        "about a technical topic, with coherent paragraphs and correct grammar."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._fasttext_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="dolma3-fasttext-quality-classifier",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_fasttext_model(self):
        model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.bin",
        )
        self._fasttext_model = fasttext.load_model(model_path)
        return self._fasttext_model

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._fasttext_model is None:
            self._load_fasttext_model()

        output_matrix = torch.tensor(self._fasttext_model.get_output_matrix())
        model = FastTextClassifierModule(output_matrix)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._fasttext_model is None:
            self._load_fasttext_model()

        vec = self._fasttext_model.get_sentence_vector(self.sample_text)
        inputs = torch.tensor(vec).unsqueeze(0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return {"sentence_vector": inputs}
