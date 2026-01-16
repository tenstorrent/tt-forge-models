# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT2 sequence classification ONNX model loader implementation
"""
import onnx
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ....tools.utils import get_file


class ModelLoader(ForgeModel):
    """GPT2 sequence classification ONNX model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.variant = "mnoukhov/gpt2-imdb-sentiment-classifier"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. Defaults to "base".

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="gpt2",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_CLASSIFICATION,
            source=ModelSource.CUSTOM,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the GPT2 sequence classification ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.variant, padding_side="left"
        )

        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/gpt2.onnx"
        # file = get_file(path)
        pt_model = AutoModelForSequenceClassification.from_pretrained(
            self.variant, return_dict=False, use_cache=False
        )
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for GPT2 sequence classification."""
        test_input = "This is a sample text from "
        input_tokens = self.tokenizer(test_input, return_tensors="pt")
        inputs = [input_tokens["input_ids"]]
        return inputs

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable text."""
        predicted_value = outputs[0].argmax(-1).item()
        return self.pt_model.config.id2label[predicted_value]
