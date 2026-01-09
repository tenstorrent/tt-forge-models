# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T5 causal language modeling ONNX model loader implementation
"""
import onnx
from transformers import AutoTokenizer

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ....tools.utils import get_file
from ...pytorch import ModelLoader as PTModelLoader


class ModelLoader(ForgeModel):
    """T5 causal language modeling ONNX model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.variant = "t5-small"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. Defaults to "base".

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="t5",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_CLASSIFICATION,
            source=ModelSource.CUSTOM,
            framework=Framework.ONNX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the T5 causal language modeling ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.variant, padding_side="left"
        )

        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/t5.onnx"
        # file = get_file(path)
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for T5 causal language modeling.

        Returns:
            torch.Tensor: Input tensor of shape [1, 3, H, W].
        """
        pt_loader = PTModelLoader()
        self.model = pt_loader.load_model()
        return pt_loader.load_inputs()
