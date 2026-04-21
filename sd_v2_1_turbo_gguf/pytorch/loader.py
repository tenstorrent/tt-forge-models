# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion v2.1 Turbo GGUF model loader implementation.

Loads GGUF-quantized UNet from gpustack/stable-diffusion-v2-1-turbo-GGUF.

SD-Turbo is a distilled version of Stable Diffusion 2.1, trained using
Adversarial Diffusion Distillation (ADD) for high-quality single-step
image generation at 512x512 resolution.

Available variants:
- Q4_0: 4-bit quantization (~2.19 GB)
- Q4_1: 4-bit quantization (~2.2 GB)
- Q8_0: 8-bit quantization (~2.32 GB)
"""

import json
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

GGUF_REPO = "gpustack/stable-diffusion-v2-1-turbo-GGUF"
BASE_PIPELINE = "stabilityai/sd-turbo"


class ModelVariant(StrEnum):
    """Available Stable Diffusion v2.1 Turbo GGUF variants."""

    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_0: "stable-diffusion-v2-1-turbo-Q4_0.gguf",
    ModelVariant.Q4_1: "stable-diffusion-v2-1-turbo-Q4_1.gguf",
    ModelVariant.Q8_0: "stable-diffusion-v2-1-turbo-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Stable Diffusion v2.1 Turbo GGUF model loader."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.Q4_1: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD_V2_1_TURBO_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized UNet for SD-Turbo.

        Diffusers' GGUF quantizer does not properly handle full SD checkpoint
        GGUF files (which use CompVis key naming and store raw quantized byte
        data rather than logical float shapes). We work around this by manually
        dequantizing with the gguf library, then converting from CompVis to
        diffusers format and loading into an empty UNet.
        """
        import gguf as gguf_lib
        from diffusers import UNet2DConditionModel
        from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Fetch the UNet config JSON without downloading full model weights.
        config_path = hf_hub_download(
            repo_id=BASE_PIPELINE, subfolder="unet", filename="config.json"
        )
        with open(config_path) as f:
            unet_config = json.load(f)

        unet = UNet2DConditionModel.from_config(unet_config)

        # Download GGUF and dequantize all tensors. The gguf library returns
        # correct logical shapes (e.g. (320, 1024)) rather than the raw byte
        # data shapes that diffusers' own GGUF quantizer incorrectly compares.
        gguf_file = _GGUF_FILES[self._variant]
        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
        reader = gguf_lib.GGUFReader(model_path)
        checkpoint = {}
        for tensor in reader.tensors:
            data = gguf_lib.dequantize(tensor.data, tensor.tensor_type)
            checkpoint[tensor.name] = torch.from_numpy(data.copy()).float()

        # Convert CompVis UNet weights (model.diffusion_model.*) to diffusers keys.
        converted_unet = convert_ldm_unet_checkpoint(checkpoint, unet_config)

        # Older SD checkpoints store proj_in/proj_out as 1x1 convolutions
        # (shape N,N,1,1) while newer diffusers uses linear layers (shape N,N).
        for key in list(converted_unet.keys()):
            if ("proj_in.weight" in key or "proj_out.weight" in key) and converted_unet[
                key
            ].dim() == 4:
                converted_unet[key] = converted_unet[key].squeeze(-1).squeeze(-1)

        unet.load_state_dict(converted_unet, strict=False)
        return unet.to(compute_dtype)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return synthetic UNet inputs for SD 2.x at 512x512 resolution.

        SD-Turbo generates 512x512 images → 64x64 latents (8x downscale).
        Cross-attention dim is 1024 (OpenCLIP ViT-H/14 text encoder).
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return {
            "sample": torch.randn(batch_size, 4, 64, 64, dtype=dtype),
            "timestep": torch.tensor([999] * batch_size),
            "encoder_hidden_states": torch.randn(batch_size, 77, 1024, dtype=dtype),
            "return_dict": False,
        }
