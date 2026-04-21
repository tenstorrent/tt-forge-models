# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NuNER ONNX model loader implementation for token classification.

Loads the pre-exported ONNX model from protectai/guishe-nuner-v1_orgs-onnx,
a RoBERTa-based NER model fine-tuned to recognize organization (ORG) entities.
"""

import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

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
    """Available NuNER ONNX token classification model variants."""

    PROTECTAI_GUISHE_NUNER_V1_ORGS_ONNX = "protectai/guishe-nuner-v1_orgs-onnx"


class ModelLoader(ForgeModel):
    """NuNER ONNX model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.PROTECTAI_GUISHE_NUNER_V1_ORGS_ONNX: ModelConfig(
            pretrained_model_name="protectai/guishe-nuner-v1_orgs-onnx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PROTECTAI_GUISHE_NUNER_V1_ORGS_ONNX

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.sample_text = (
            "Apple and Google announced a new partnership with CNN at the conference."
        )
        self.max_length = 128

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NuNER",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the pre-exported NuNER ONNX model from Hugging Face.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        onnx_path = hf_hub_download(
            repo_id=pretrained_model_name, filename="model.onnx"
        )
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, **kwargs):
        """Prepare tokenized inputs for the NuNER ONNX model."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
