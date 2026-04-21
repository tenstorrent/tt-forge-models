# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Real-ESRGAN model loader implementation for image super-resolution
"""

import torch
from dataclasses import dataclass
from typing import Optional
from safetensors.torch import load_file
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
from .src.rrdbnet import RRDBNet


@dataclass
class RealESRGANConfig(ModelConfig):
    """Configuration specific to Real-ESRGAN models."""

    filename: str = "RealESRGAN_x4plus.safetensors"


class ModelVariant(StrEnum):
    """Available Real-ESRGAN model variants."""

    X4PLUS = "x4plus"
    X4PLUS_SDCPP_GGUF = "x4plus_sdcpp_gguf"


class ModelLoader(ForgeModel):
    """Real-ESRGAN model loader implementation for image super-resolution tasks."""

    _VARIANTS = {
        ModelVariant.X4PLUS: RealESRGANConfig(
            pretrained_model_name="Comfy-Org/Real-ESRGAN_repackaged",
            filename="RealESRGAN_x4plus.safetensors",
        ),
        ModelVariant.X4PLUS_SDCPP_GGUF: RealESRGANConfig(
            pretrained_model_name="wbruna/upscalers-sdcpp-gguf",
            filename="RealESRGAN_x4plus.gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.X4PLUS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Real-ESRGAN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _load_gguf_state_dict(weights_path: str) -> dict:
        """Load tensors from a GGUF file into a PyTorch state_dict.

        The stable-diffusion.cpp convert tool preserves the original PyTorch
        state_dict keys, so tensors can be mapped directly by name.
        """
        import gguf

        reader = gguf.GGUFReader(weights_path)
        state_dict = {}
        for tensor in reader.tensors:
            array = tensor.data
            if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                array = array.view("float16")
            elif tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                array = array.view("float32")
            shape = tuple(reversed([int(d) for d in tensor.shape]))
            state_dict[tensor.name] = torch.from_numpy(array.reshape(shape).copy())
        return state_dict

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Real-ESRGAN RRDBNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Real-ESRGAN model instance.
        """
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        weights_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self._variant_config.filename,
        )

        if self._variant_config.filename.endswith(".gguf"):
            state_dict = self._load_gguf_state_dict(weights_path)
        else:
            state_dict = load_file(weights_path)

        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Real-ESRGAN model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the input tensor.

        Returns:
            torch.Tensor: Input image tensor of shape [batch, 3, 64, 64].
        """
        inputs = torch.randn(batch_size, 3, 64, 64)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
