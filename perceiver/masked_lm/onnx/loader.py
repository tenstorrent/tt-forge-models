# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Perceiver image classification ONNX model loader implementation
"""
import onnx
from transformers import PerceiverTokenizer

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
    """Perceiver image classification ONNX model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.variant = "deepmind/language-perceiver"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. Defaults to "base".

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="perceiver",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_CLASSIFICATION,
            source=ModelSource.CUSTOM,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the Perceiver image classification ONNX model.

        Keyword Args:
            path (str): Absolute path to the ONNX file. If not provided,
                defaults to the project's onnx_dir perceiver file.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.tokenizer = PerceiverTokenizer.from_pretrained(self.variant)

        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/perceiver.onnx"
        # file = get_file(path)
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for Perceiver image classification.

        Keyword Args:
            image_path (str): Optional path or URL to an image. If not provided,
                a sample image from the web will be used.
            resolution (tuple): Optional (H, W) resolution. Defaults to (224, 224).

        Returns:
            torch.Tensor: Input tensor of shape [1, 3, H, W].
        """
        text = "This is an incomplete sentence where some words are missing."
        encoding = self.tokenizer(text, padding="max_length", return_tensors="pt")
        encoding.input_ids[0, 52:61] = self.tokenizer.mask_token_id
        inputs = [encoding.input_ids, encoding.attention_mask]
        return inputs

    def decode_output(self, outputs, top_k: int = 5):
        """Decode logits to top-k class indices.

        Args:
            outputs: Model outputs (logits tensor or numpy array).
            top_k: Number of top classes to return.

        Returns:
            list[int]: Indices of top-k predicted classes.
        """
        logits = outputs[0]
        masked_tokens_predictions = logits[0, 51:61].argmax(dim=-1)
        return self.tokenizer.decode(masked_tokens_predictions)
