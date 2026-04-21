# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chrome on-device models loader implementation.

dejanseo/chrome_models is a collection of Google Chrome's on-device machine
learning models distributed as TensorFlow Lite files. Each model corresponds
to a Chrome "optimization target" such as LANGUAGE_DETECTION, PAGE_TOPICS_V2
or PAGE_VISIBILITY. This loader downloads a selected TFLite model and wraps
it in a PyTorch-compatible interface for use with the test harness.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from huggingface_hub import hf_hub_download

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Chrome on-device model variants."""

    LANGUAGE_DETECTION = "language_detection"


class ChromeTFLiteWrapper(nn.Module):
    """PyTorch wrapper around a Chrome on-device TFLite model."""

    def __init__(self, tflite_model_path: str):
        super().__init__()
        import ai_edge_litert as tflite

        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def forward(self, *inputs: torch.Tensor):
        for detail, tensor in zip(self.input_details, inputs):
            array = tensor.detach().cpu().numpy().astype(detail["dtype"])
            self.interpreter.resize_tensor_input(detail["index"], array.shape)
        self.interpreter.allocate_tensors()

        for detail, tensor in zip(self.input_details, inputs):
            array = tensor.detach().cpu().numpy().astype(detail["dtype"])
            self.interpreter.set_tensor(detail["index"], array)

        self.interpreter.invoke()

        outputs = [
            torch.from_numpy(self.interpreter.get_tensor(detail["index"]).copy())
            for detail in self.output_details
        ]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class ModelLoader(ForgeModel):
    """Loader for Chrome on-device TFLite models from dejanseo/chrome_models."""

    _VARIANTS = {
        ModelVariant.LANGUAGE_DETECTION: ModelConfig(
            pretrained_model_name="dejanseo/chrome_models",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LANGUAGE_DETECTION

    _TFLITE_FILES = {
        ModelVariant.LANGUAGE_DETECTION: "2/model.tflite",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="chrome_models",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_model(self):
        if self._model is None:
            repo_id = self._variant_config.pretrained_model_name
            filename = self._TFLITE_FILES[self._variant]
            tflite_path = hf_hub_download(repo_id=repo_id, filename=filename)
            self._model = ChromeTFLiteWrapper(tflite_path)
            self._model.eval()
        return self._model

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Chrome TFLite model wrapped as a PyTorch module."""
        return self._ensure_model()

    def load_inputs(self, dtype_override=None):
        """Load sample inputs matching the TFLite model's input signature."""
        model = self._ensure_model()

        _NUMPY_TO_TORCH_DTYPE = {
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.float16: torch.float16,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
            np.bool_: torch.bool,
        }

        inputs = []
        for detail in model.input_details:
            shape = tuple(int(d) if d > 0 else 1 for d in detail["shape"])
            np_dtype = detail["dtype"]
            torch_dtype = _NUMPY_TO_TORCH_DTYPE.get(np_dtype, torch.float32)
            if torch_dtype.is_floating_point:
                tensor = torch.rand(shape, dtype=torch_dtype)
            else:
                tensor = torch.zeros(shape, dtype=torch_dtype)
            inputs.append(tensor)

        return inputs
