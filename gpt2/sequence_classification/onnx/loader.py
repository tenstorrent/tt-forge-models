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
        self.variant = "gpt2"

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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.variant, return_dict=False, use_cache=False
        )
        self.model.eval()
        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/gpt2.onnx"
        # file = get_file(path)
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for GPT2 sequence classification.

        Keyword Args:
            image_path (str): Optional path or URL to an image. If not provided,
                a sample image from the web will be used.
            resolution (tuple): Optional (H, W) resolution. Defaults to (224, 224).

        Returns:
            torch.Tensor: Input tensor of shape [1, 3, H, W].
        """
        test_input = "This is a sample text from "
        input_tokens = self.tokenizer(test_input, return_tensors="pt")
        inputs = [input_tokens["input_ids"]]
        return inputs

    def decode_output(self, outputs, top_k: int = 5):
        """Decode logits to top-k class indices.

        Args:
            outputs: Model outputs (logits tensor or numpy array).
            top_k: Number of top classes to return.

        Returns:
            list[int]: Indices of top-k predicted classes.
        """
        predicted_value = outputs[0].argmax(-1).item()
        return self.model.config.id2label[predicted_value]
