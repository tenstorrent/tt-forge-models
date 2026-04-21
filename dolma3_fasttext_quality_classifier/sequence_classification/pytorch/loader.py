# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""allenai/dolma3-fasttext-quality-classifier model loader for sequence classification.

The Dolma 3 quality classifier is a fastText binary classifier that labels a
document as either high or low quality. fastText models are not native PyTorch
modules, so we rebuild the linear classification head as a torch module and
feed it pre-computed sentence vectors from the underlying fastText model.
"""

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
    """Available Dolma 3 fastText quality classifier variants."""

    DOLMA3_FASTTEXT_QUALITY_CLASSIFIER = "dolma3-fasttext-quality-classifier"


class FastTextClassifierModule(torch.nn.Module):
    """PyTorch wrapper around the linear head of a fastText classifier."""

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
    """allenai/dolma3-fasttext-quality-classifier model loader."""

    _VARIANTS = {
        ModelVariant.DOLMA3_FASTTEXT_QUALITY_CLASSIFIER: ModelConfig(
            pretrained_model_name="allenai/dolma3-fasttext-quality-classifier",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOLMA3_FASTTEXT_QUALITY_CLASSIFIER

    sample_text = (
        "Transformer architectures have substantially advanced the state of "
        "natural language understanding tasks over recent years."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._fasttext_model = None
        self._labels = None

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
        self._labels = self._fasttext_model.get_labels()
        return self._fasttext_model

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._fasttext_model is None:
            self._load_fasttext_model()

        output_matrix = torch.from_numpy(
            self._fasttext_model.get_output_matrix()
        ).float()
        model = FastTextClassifierModule(output_matrix)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._fasttext_model is None:
            self._load_fasttext_model()

        sentence_vector = (
            torch.from_numpy(self._fasttext_model.get_sentence_vector(self.sample_text))
            .float()
            .unsqueeze(0)
        )

        if dtype_override is not None:
            sentence_vector = sentence_vector.to(dtype_override)

        return {"sentence_vector": sentence_vector}

    def decode_output(self, co_out):
        if self._labels is None:
            raise RuntimeError("Model must be loaded before decoding outputs.")

        probs = co_out[0].detach().cpu().float()
        predicted_idx = int(torch.argmax(probs).item())
        predicted_label = self._labels[predicted_idx]
        print(
            f"Predicted quality label: {predicted_label} "
            f"(prob={probs[predicted_idx].item():.4f})"
        )
