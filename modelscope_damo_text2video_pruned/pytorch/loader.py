# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
kabachuha/modelscope-damo-text2video-pruned-weights model loader.

This repository hosts a pruned fp16 copy of the original ModelScope DAMO
text-to-video synthesis weights. Files are published in raw ModelScope
format (not diffusers layout):
- text2video_pytorch_model.pth: UNet3D diffusion model weights
- VQGAN_autoencoder.pth: VAE weights
- open_clip_pytorch_model.bin: OpenCLIP text encoder weights
- configuration.json: ModelScope pipeline configuration

The architecture matches the diffusers mirror ali-vilab/text-to-video-ms-1.7b,
so the loader instantiates a diffusers UNet3DConditionModel with that config
and loads the pruned checkpoint.

Repository:
- https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download

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


class ModelVariant(StrEnum):
    """Available modelscope-damo-text2video-pruned-weights variants."""

    PRUNED_FP16 = "pruned_fp16"


class ModelLoader(ForgeModel):
    """Loader for kabachuha/modelscope-damo-text2video-pruned-weights."""

    _VARIANTS = {
        ModelVariant.PRUNED_FP16: ModelConfig(
            pretrained_model_name="kabachuha/modelscope-damo-text2video-pruned-weights",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PRUNED_FP16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="modelscope_damo_text2video_pruned",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ModelScope DAMO text-to-video UNet3D with pruned weights.

        Builds a diffusers UNet3DConditionModel with the architecture config
        from ali-vilab/text-to-video-ms-1.7b (the diffusers-format port of the
        same ModelScope model), then loads the pruned fp16 checkpoint from the
        kabachuha repository. The raw checkpoint uses ModelScope's key scheme
        rather than diffusers', so state_dict loading is non-strict.

        Args:
            dtype_override: Optional torch.dtype to override the model dtype.

        Returns:
            torch.nn.Module: The UNet3DConditionModel instance.
        """
        from diffusers import UNet3DConditionModel

        repo_id = self._variant_config.pretrained_model_name

        unet = UNet3DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            up_block_types=(
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
            ),
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=1024,
            attention_head_dim=64,
            norm_num_groups=32,
            norm_eps=1e-5,
        )

        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename="text2video_pytorch_model.pth"
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        unet.load_state_dict(state_dict, strict=False)
        unet.eval()

        if dtype_override is not None:
            unet = unet.to(dtype=dtype_override)

        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Synthetic inputs for the UNet3D forward pass.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Dictionary of input tensors.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        num_frames = 2
        height = 8
        width = 8

        sample = torch.randn(batch_size, 4, num_frames, height, width, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, 16, 1024, dtype=dtype)
        timestep = torch.tensor([0], dtype=torch.long).expand(batch_size)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
