# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QuantStack/Qwen-Image-GGUF model loader implementation.

Loads GGUF-quantized diffusion transformer variants from
QuantStack/Qwen-Image-GGUF. Uses the upstream Qwen/Qwen-Image diffusers
config for model construction.

Available variants:
- Q4_K_M: 4-bit quantization (medium, 13.1 GB)
- Q8_0: 8-bit quantization (21.8 GB)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageTransformer2DModel
from huggingface_hub import hf_hub_download

from ...base import ForgeModel


def _patch_gguf_loader():
    """Patch diffusers 0.37.1 load_gguf_checkpoint to dequantize all tensors on load.

    In diffusers 0.37.1, GGUF quantized tensors are loaded as GGUFParameter with
    raw uint8 byte shapes, but load_model_dict_into_meta checks raw shape vs model
    shape without a GGUF quantizer configured (requires explicit quantization_config).
    BF16 tensors are also incorrectly treated as quantized, doubling the apparent
    last dimension. This patch dequantizes all tensors at load time so plain
    float tensors with correct shapes are returned.
    """
    from gguf import GGUFReader, GGMLQuantizationType
    from diffusers.models import model_loading_utils
    from diffusers.quantizers.gguf.utils import (
        GGUFParameter,
        SUPPORTED_GGUF_QUANT_TYPES,
        dequantize_gguf_tensor,
    )

    _UNQUANTIZED = {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
    }

    def _patched_load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
        reader = GGUFReader(gguf_checkpoint_path)
        parsed_parameters = {}
        for tensor in reader.tensors:
            name = tensor.name
            quant_type = tensor.tensor_type
            if quant_type == GGMLQuantizationType.BF16:
                weights = torch.from_numpy(tensor.data.copy()).view(torch.bfloat16)
            elif quant_type in _UNQUANTIZED:
                weights = torch.from_numpy(tensor.data.copy())
            else:
                if quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
                    supported = "\n".join(str(t) for t in SUPPORTED_GGUF_QUANT_TYPES)
                    raise ValueError(
                        f"{name} has unsupported quantization type: {quant_type}\n\n"
                        f"Supported types:\n{supported}"
                    )
                raw = torch.from_numpy(tensor.data.copy())
                weights = dequantize_gguf_tensor(
                    GGUFParameter(raw, quant_type=quant_type)
                )
            parsed_parameters[name] = weights
        return parsed_parameters

    model_loading_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

REPO_ID = "QuantStack/Qwen-Image-GGUF"
CONFIG_REPO = "Qwen/Qwen-Image"

_GGUF_FILES = {
    "Q4_K_M": "Qwen_Image-Q4_K_M.gguf",
    "Q8_0": "Qwen_Image-Q8_0.gguf",
}


class ModelVariant(StrEnum):
    """Available Qwen-Image GGUF quantization variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """QuantStack/Qwen-Image-GGUF model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.float32
    ) -> QwenImageTransformer2DModel:
        """Load diffusion transformer from GGUF file."""
        gguf_filename = _GGUF_FILES[str(self._variant.value)]

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=gguf_filename,
        )

        _patch_gguf_loader()
        self._transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Qwen-Image GGUF diffusion transformer."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = kwargs.get("batch_size", 1)

        # From Qwen-Image config: in_channels=64 (img_in linear input dimension)
        img_dim = 64
        # joint_attention_dim from config = 3584
        text_dim = 3584
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
