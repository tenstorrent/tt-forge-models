# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Fun InP GGUF model loader implementation for video generation
"""
from typing import Optional

import torch
from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
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

GGUF_REPO = "QuantStack/Wan2.2-Fun-A14B-InP-GGUF"


class ModelVariant(StrEnum):
    """Available Wan 2.2 Fun InP GGUF model variants."""

    A14B_HIGHNOISE_Q4_K_M = "A14B_HighNoise_Q4_K_M"


_GGUF_FILES = {
    ModelVariant.A14B_HIGHNOISE_Q4_K_M: "HighNoise/Wan2.2-Fun-A14B-InP_HighNoise-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Fun InP GGUF model loader for video generation tasks."""

    _VARIANTS = {
        ModelVariant.A14B_HIGHNOISE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.A14B_HIGHNOISE_Q4_K_M

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
        import diffusers.loaders.single_file_model as _sfm

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_filename)

        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        # GGUF models loaded with low_cpu_mem_usage=True (default) may leave
        # some non-quantized params on the meta device if they aren't in the
        # GGUF file's tensor list.  accelerate's dispatch_model then calls
        # model.to(device) which fails on meta tensors.  Patch dispatch_model
        # to materialise any remaining meta params before dispatching.
        _orig_dispatch = _sfm.dispatch_model

        def _safe_dispatch(model, device_map=None, **kw):
            if device_map:
                target = next(iter(device_map.values()))
                for mod in model.modules():
                    for pname, param in list(mod.named_parameters(recurse=False)):
                        if param.device.type == "meta":
                            mod.register_parameter(
                                pname,
                                torch.nn.Parameter(
                                    torch.empty(
                                        param.shape, dtype=param.dtype, device=target
                                    ),
                                    requires_grad=param.requires_grad,
                                ),
                            )
                    for bname, buf in list(mod.named_buffers(recurse=False)):
                        if isinstance(buf, torch.Tensor) and buf.device.type == "meta":
                            mod.register_buffer(
                                bname,
                                torch.empty(buf.shape, dtype=buf.dtype, device=target),
                            )
            return _orig_dispatch(model, device_map=device_map, **kw)

        _sfm.dispatch_model = _safe_dispatch
        try:
            self.transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            _sfm.dispatch_model = _orig_dispatch

        # TT-XLA's __torch_function__ intercepts F.linear before GGUFLinear can
        # dequantize, causing a Byte vs BFloat16 dtype mismatch. Convert all
        # GGUFLinear layers to standard nn.Linear with dequantized weights.
        from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear

        self.transformer = _dequantize_gguf_and_restore_linear(self.transformer)
        self.transformer = self.transformer.to(compute_dtype)

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
