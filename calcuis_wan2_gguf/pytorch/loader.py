#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
calcuis/wan2-gguf model loader implementation.

Loads GGUF-quantized Wan 2.2 diffusion transformers from the
calcuis/wan2-gguf repository. The repo packages Wan 2.2 Animate and
Wan 2.2 I2V (high-noise and low-noise experts) variants for ComfyUI use.

Available variants:
- WAN22_ANIMATE_Q4_K_M: Wan 2.2 Animate 14B, Q4_K_M quantization
- WAN22_I2V_HIGH_NOISE_Q4_K_M: Wan 2.2 I2V 14B high-noise expert, Q4_K_M
- WAN22_I2V_LOW_NOISE_Q4_K_M: Wan 2.2 I2V 14B low-noise expert, Q4_K_M
"""

import json
import os
import tempfile
from typing import Any, Optional

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

GGUF_REPO = "calcuis/wan2-gguf"

# Base Wan 2.1 I2V diffusers config repo — we load its config and strip
# Wan 2.1-specific fields that are absent from the Wan 2.2 GGUF.
WAN21_I2V_DIFFUSERS_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# Small spatial dimensions for compile-only testing
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available calcuis/wan2-gguf variants."""

    WAN22_ANIMATE_Q4_K_M = "2.2_Animate_Q4_K_M"
    WAN22_I2V_HIGH_NOISE_Q4_K_M = "2.2_I2V_HighNoise_Q4_K_M"
    WAN22_I2V_LOW_NOISE_Q4_K_M = "2.2_I2V_LowNoise_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.WAN22_ANIMATE_Q4_K_M: "wan2.2-animate-14b-q4_k_m.gguf",
    ModelVariant.WAN22_I2V_HIGH_NOISE_Q4_K_M: "wan2.2-i2v-14b-high-noise-q4_k_m.gguf",
    ModelVariant.WAN22_I2V_LOW_NOISE_Q4_K_M: "wan2.2-i2v-14b-low-noise-q4_k_m.gguf",
}

# Wan 2.2 I2V GGUFs omit the image-conditioning components present in
# the Wan 2.1 I2V diffusers config.  Removing these fields from the
# config prevents 208 parameters from being created as meta tensors,
# which would cause dispatch_model to fail with NotImplementedError.
_WAN22_I2V_STRIP_CONFIG_KEYS = ("added_kv_proj_dim", "image_dim")

_I2V_VARIANTS = {
    ModelVariant.WAN22_I2V_HIGH_NOISE_Q4_K_M,
    ModelVariant.WAN22_I2V_LOW_NOISE_Q4_K_M,
}


class ModelLoader(ForgeModel):
    """calcuis/wan2-gguf model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_ANIMATE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_I2V_HIGH_NOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.WAN22_I2V_LOW_NOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_ANIMATE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CALCUIS_WAN2_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized Wan 2.2 transformer.

        Wan 2.2 I2V GGUFs do not include the image-conditioning modules
        (added_kv_proj_dim, image_dim) from the Wan 2.1 I2V config.
        We strip those fields before constructing the model so that
        dispatch_model does not encounter meta tensors.
        """
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from diffusers import (
            GGUFQuantizationConfig,
            WanTransformer3DModel,
        )
        from huggingface_hub import hf_hub_download

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        if self._variant in _I2V_VARIANTS:
            # Load Wan 2.1 I2V config, strip Wan 2.1-specific fields absent
            # from the Wan 2.2 GGUF, then write to a temp dir for from_single_file.
            base_config = WanTransformer3DModel.load_config(
                pretrained_model_name_or_path=WAN21_I2V_DIFFUSERS_REPO,
                subfolder="transformer",
            )
            wan22_config = {
                k: v
                for k, v in base_config.items()
                if k not in _WAN22_I2V_STRIP_CONFIG_KEYS
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = os.path.join(tmpdir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(wan22_config, f)
                self._transformer = WanTransformer3DModel.from_single_file(
                    gguf_path,
                    quantization_config=quantization_config,
                    torch_dtype=compute_dtype,
                    config=tmpdir,
                )
        else:
            self._transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )

        return self._transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare tensor inputs for the WanTransformer3DModel forward pass."""
        if self._transformer is None:
            self.load_model()

        dtype = torch.bfloat16
        config = self._transformer.config

        return {
            "hidden_states": torch.randn(
                1,
                config.in_channels,
                TRANSFORMER_NUM_FRAMES,
                TRANSFORMER_HEIGHT,
                TRANSFORMER_WIDTH,
                dtype=dtype,
            ),
            "encoder_hidden_states": torch.randn(
                1,
                TRANSFORMER_TEXT_SEQ_LEN,
                config.text_dim,
                dtype=dtype,
            ),
            "timestep": torch.tensor([500], dtype=torch.long),
            "return_dict": False,
        }
