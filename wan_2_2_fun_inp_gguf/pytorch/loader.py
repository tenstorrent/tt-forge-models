# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Fun InP GGUF model loader implementation for video generation
"""

import torch
from typing import Optional

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
    """Available Wan 2.2 Fun InP GGUF model variants."""

    A14B_HIGHNOISE_Q4_K_M = "A14B_HighNoise_Q4_K_M"


class ModelLoader(ForgeModel):
    """Wan 2.2 Fun InP GGUF model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.A14B_HIGHNOISE_Q4_K_M: ModelConfig(
            pretrained_model_name="QuantStack/Wan2.2-Fun-A14B-InP-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.A14B_HIGHNOISE_Q4_K_M

    GGUF_FILE = "HighNoise/Wan2.2-Fun-A14B-InP_HighNoise-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wan 2.2 Fun InP GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        import os
        import shutil

        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import HF_HUB_CACHE

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        # Fall back to /tmp when the standard HF cache location lacks space.
        gguf_size_bytes = 10 * 1024**3
        try:
            free_bytes = shutil.disk_usage(HF_HUB_CACHE).free
        except FileNotFoundError:
            free_bytes = 0
        cache_dir = (
            HF_HUB_CACHE if free_bytes >= gguf_size_bytes else "/tmp/hf_gguf_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)

        gguf_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
            cache_dir=cache_dir,
        )

        # WanRotaryPosEmbed registers freqs_cos/freqs_sin as non-persistent buffers
        # during __init__. With init_empty_weights (used by low_cpu_mem_usage=True)
        # these become meta tensors that are absent from the GGUF state dict, causing
        # dispatch_model to fail. Patch dispatch_model to materialise such meta buffers
        # as zeros so loading can complete, then re-init the rope module with real values.
        import diffusers.loaders.single_file_model as _sfm
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        _orig_dispatch = getattr(_sfm, "dispatch_model", None)

        def _patched_dispatch(model, **dkw):
            # Materialise any meta parameters/buffers that were absent from the
            # GGUF state dict (e.g. biases set to None in the checkpoint).
            for name, param in list(model.named_parameters()):
                if param.is_meta:
                    parts = name.rsplit(".", 1)
                    parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
                    parent._parameters[parts[-1]] = torch.nn.Parameter(
                        torch.zeros(param.shape, dtype=param.dtype)
                    )
            for name, buf in list(model.named_buffers()):
                if buf is not None and buf.is_meta:
                    parts = name.rsplit(".", 1)
                    parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
                    parent._buffers[parts[-1]] = torch.zeros(buf.shape, dtype=buf.dtype)
            if _orig_dispatch is not None:
                return _orig_dispatch(model, **dkw)

        if _orig_dispatch is not None:
            _sfm.dispatch_model = _patched_dispatch
        try:
            self.transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            if _orig_dispatch is not None:
                _sfm.dispatch_model = _orig_dispatch

        # Replace the rope module so freqs_cos/freqs_sin are properly computed.
        old_rope = self.transformer.rope
        self.transformer.rope = WanRotaryPosEmbed(
            old_rope.attention_head_dim,
            old_rope.patch_size,
            old_rope.max_seq_len,
        )

        # GGUFLinear stores weights as packed byte tensors and relies on
        # GGUFParameter.__torch_function__ to dequantize on-the-fly. The TT-XLA
        # __torch_function__ override intercepts F.linear before GGUF can
        # dequantize, causing a dtype mismatch. Convert all GGUFLinear modules
        # back to plain nn.Linear with float weights before compilation.
        from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear

        _dequantize_gguf_and_restore_linear(self.transformer)
        # Remove quantizer markers so diffusers' .to() guard doesn't block dtype cast.
        self.transformer.hf_quantizer = None
        self.transformer.is_quantized = False
        self.transformer.to(compute_dtype)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Video dimensions: (batch, channels, frames, height, width)
        num_frames = 1
        height = 32
        width = 32
        in_channels = config.in_channels

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)

        # Text encoder hidden states
        text_dim = config.text_dim
        seq_len = 64
        encoder_hidden_states = torch.randn(batch_size, seq_len, text_dim, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        return inputs
