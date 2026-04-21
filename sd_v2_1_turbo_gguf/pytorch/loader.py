# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion v2.1 Turbo GGUF model loader implementation.

Loads GGUF-quantized UNet from gpustack/stable-diffusion-v2-1-turbo-GGUF
and builds a text-to-image pipeline using the base stabilityai/sd-turbo model.

SD-Turbo is a distilled version of Stable Diffusion 2.1, trained using
Adversarial Diffusion Distillation (ADD) for high-quality single-step
image generation at 512x512 resolution.

Available variants:
- Q4_0: 4-bit quantization (~2.19 GB)
- Q4_1: 4-bit quantization (~2.2 GB)
- Q8_0: 8-bit quantization (~2.32 GB)
"""

from typing import Optional

import torch

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
        self.pipeline = None

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
        """Load the GGUF-quantized UNet and build the SD-Turbo pipeline.

        Diffusers' GGUF quantizer does not properly handle full SD checkpoint
        GGUF files (which use CompVis key naming and store raw quantized byte
        data). We work around this by manually dequantizing with the gguf
        library, then converting from CompVis to diffusers format.
        """
        import gguf as gguf_lib
        from diffusers import StableDiffusionPipeline
        from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        # Load the base pipeline to obtain all non-UNet components and the UNet config.
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            torch_dtype=compute_dtype,
        )

        # Dequantize all tensors from the full-checkpoint GGUF using the gguf
        # library, which handles Q4_0/Q4_1/Q8_0 etc. and returns correct shapes.
        reader = gguf_lib.GGUFReader(model_path)
        checkpoint = {}
        for tensor in reader.tensors:
            data = gguf_lib.dequantize(tensor.data, tensor.tensor_type)
            checkpoint[tensor.name] = torch.from_numpy(data.copy()).float()

        # Convert the extracted CompVis UNet weights (model.diffusion_model.*)
        # to diffusers parameter names.
        unet_config = dict(self.pipeline.unet.config)
        converted_unet = convert_ldm_unet_checkpoint(checkpoint, unet_config)
        self.pipeline.unet.load_state_dict(converted_unet, strict=False)
        self.pipeline.unet = self.pipeline.unet.to(compute_dtype)

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for SD-Turbo.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        ] * batch_size
        return prompt
