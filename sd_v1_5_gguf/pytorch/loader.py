# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion v1.5 GGUF model loader implementation.

Loads a GGUF-quantized UNet from gpustack/stable-diffusion-v1-5-GGUF and
builds a text-to-image pipeline using the base sd-legacy/stable-diffusion-v1-5
model for the remaining components (text encoder, tokenizer, VAE, scheduler).

Available variants:
- FP16: Full precision (~2.13 GB)
- Q4_0: 4-bit quantization (~1.75 GB)
- Q4_1: 4-bit quantization (~1.76 GB)
- Q8_0: 8-bit quantization (~1.88 GB)
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

GGUF_REPO = "gpustack/stable-diffusion-v1-5-GGUF"
BASE_PIPELINE = "sd-legacy/stable-diffusion-v1-5"


def _dequantize_q8_0(data: torch.Tensor) -> torch.Tensor:
    """Dequantize a Q8_0 GGUF uint8 tensor to float32.

    Q8_0 blocks: [float16 scale (2 bytes), int8 values (32 bytes)] = 34 bytes/block.
    Diffusers 0.37.1 leaves norm weights as raw uint8 bytes instead of dequantizing them.
    """
    n_blocks = data.numel() // 34
    raw = data.reshape(n_blocks, 34)
    scale = raw[:, :2].contiguous().view(torch.float16).float()
    q_values = raw[:, 2:].contiguous().to(torch.int8).float()
    return (q_values * scale.unsqueeze(1)).reshape(n_blocks * 32)


def _fix_gguf_norm_weights(unet: torch.nn.Module, compute_dtype: torch.dtype) -> None:
    """Fix norm layer weights left as raw Q8_0 bytes by diffusers 0.37.1."""
    import torch.nn as nn

    with torch.no_grad():
        for module in unet.modules():
            if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                if module.weight is not None and module.weight.dtype == torch.uint8:
                    module.weight.data = _dequantize_q8_0(module.weight.data).to(
                        compute_dtype
                    )
                if module.bias is not None and module.bias.dtype == torch.uint8:
                    module.bias.data = _dequantize_q8_0(module.bias.data).to(
                        compute_dtype
                    )


class ModelVariant(StrEnum):
    """Available Stable Diffusion v1.5 GGUF variants."""

    FP16 = "FP16"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.FP16: "stable-diffusion-v1-5-FP16.gguf",
    ModelVariant.Q4_0: "stable-diffusion-v1-5-Q4_0.gguf",
    ModelVariant.Q4_1: "stable-diffusion-v1-5-Q4_1.gguf",
    ModelVariant.Q8_0: "stable-diffusion-v1-5-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Stable Diffusion v1.5 GGUF model loader."""

    _VARIANTS = {
        ModelVariant.FP16: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
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
            model="SD_V1_5_GGUF",
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
        """Load the GGUF-quantized UNet and build the SD v1.5 pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized UNet,
        then constructs the StableDiffusionPipeline with the base model's
        other components. Returns the UNet as a torch.nn.Module.
        """
        from diffusers import (
            GGUFQuantizationConfig,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        local_gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
        unet = UNet2DConditionModel.from_single_file(
            local_gguf_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            unet=unet,
            torch_dtype=compute_dtype,
        )

        _fix_gguf_norm_weights(self.pipeline.unet, compute_dtype)
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare preprocessed tensor inputs for the UNet.

        Encodes a text prompt via the pipeline's text encoder and returns
        UNet-ready latent tensors.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype
        prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[
                0
            ].to(dtype)

        in_channels = self.pipeline.unet.config.in_channels
        sample_size = self.pipeline.unet.config.sample_size
        latent_sample = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)

        return [latent_sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output) -> torch.Tensor:
        """Unpack UNet output to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
