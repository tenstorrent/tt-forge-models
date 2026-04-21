# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
calcuis/ltxv0.9.6-gguf model loader for tt_forge_models.

calcuis/ltxv0.9.6-gguf is a GGUF-quantized conversion of the Lightricks/LTX-Video
0.9.6 text-to-video model (2B parameter DiT). Available in dev and distilled
flavors across multiple quantization levels (Q2_K through Q8_0, plus BF16/F16/F32).

Repository: https://huggingface.co/calcuis/ltxv0.9.6-gguf
"""

from typing import Any, Optional

import torch
from diffusers import GGUFQuantizationConfig, LTXVideoTransformer3DModel
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

REPO_ID = "calcuis/ltxv0.9.6-gguf"
BASE_REPO = "Lightricks/LTX-Video"


class ModelVariant(StrEnum):
    """Available calcuis/ltxv0.9.6-gguf quantization variants."""

    LTXV_0_9_6_DEV_Q4_0 = "0.9.6_dev_Q4_0"
    LTXV_0_9_6_DISTILLED_Q4_0 = "0.9.6_distilled_Q4_0"


# GGUF filenames within the repository
_GGUF_FILES = {
    ModelVariant.LTXV_0_9_6_DEV_Q4_0: "ltxv-2b-0.9.6-dev-q4_0.gguf",
    ModelVariant.LTXV_0_9_6_DISTILLED_Q4_0: "ltxv-2b-0.9.6-distilled-q4_0.gguf",
}


class ModelLoader(ForgeModel):
    """Loader for calcuis/ltxv0.9.6-gguf quantized LTX-Video transformer.

    Loads the LTXVideoTransformer3DModel from a single GGUF file using
    diffusers' GGUFQuantizationConfig.
    """

    _VARIANTS = {
        ModelVariant.LTXV_0_9_6_DEV_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.LTXV_0_9_6_DISTILLED_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LTXV_0_9_6_DEV_Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer: Optional[LTXVideoTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTXV 0.9.6 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=gguf_filename,
        )

        self._transformer = LTXVideoTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            self.load_model(dtype_override=dtype)

        config = self._transformer.config

        batch_size = 1
        latent_num_frames = 2
        latent_height = 2
        latent_width = 2
        video_seq_len = latent_num_frames * latent_height * latent_width

        hidden_states = torch.randn(
            batch_size, video_seq_len, config.in_channels, dtype=dtype
        )

        caption_channels = config.caption_channels
        encoder_hidden_states = torch.randn(
            batch_size, 8, caption_channels, dtype=dtype
        )
        encoder_attention_mask = torch.ones(batch_size, 8, dtype=dtype)

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "timestep": timestep,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output
