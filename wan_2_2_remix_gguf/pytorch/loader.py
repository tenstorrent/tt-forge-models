#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Remix GGUF model loader implementation.

Loads GGUF-quantized Wan 2.2 Remix transformers from community
re-uploads (BigDannyPt/Wan-2.2-Remix-GGUF and
huchukato/Wan2.2-Remix-I2V-v2.1-GGUF) and builds text-to-video or
image-to-video pipelines.

The Wan 2.2 Remix is a community fine-tune of the Wan 2.2 14B model
supporting both text-to-video (T2V) and image-to-video (I2V) generation.
Each mode has high-noise and low-noise expert variants following the
Mixture-of-Experts (MoE) architecture.

Available variants:
- WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: T2V high-noise expert v2.0, Q4_K_M
- WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: I2V high-noise expert v3.0, Q4_K_M
- WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: I2V high-noise expert v2.1, Q4_K_M
- WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: I2V low-noise expert v2.1, Q4_K_M
"""

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

BIGDANNYPT_GGUF_REPO = "BigDannyPt/Wan-2.2-Remix-GGUF"
HUCHUKATO_I2V_V2_1_GGUF_REPO = "huchukato/Wan2.2-Remix-I2V-v2.1-GGUF"
T2V_BASE_PIPELINE = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
I2V_BASE_PIPELINE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.2 Remix GGUF variants."""

    WAN22_REMIX_T2V_HIGH_V2_Q4_K_M = "2.2_Remix_T2V_High_v2.0_Q4_K_M"
    WAN22_REMIX_I2V_HIGH_V3_Q4_K_M = "2.2_Remix_I2V_High_v3.0_Q4_K_M"
    WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M = "2.2_Remix_I2V_High_v2.1_Q4_K_M"
    WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M = "2.2_Remix_I2V_Low_v2.1_Q4_K_M"


_GGUF_REPOS = {
    ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: BIGDANNYPT_GGUF_REPO,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: BIGDANNYPT_GGUF_REPO,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: HUCHUKATO_I2V_V2_1_GGUF_REPO,
    ModelVariant.WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: HUCHUKATO_I2V_V2_1_GGUF_REPO,
}

_GGUF_FILES = {
    ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: "T2V/v2.0/High/wan22RemixT2VI2V_t2vHighV20-Q4_K_M.gguf",
    ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: "I2V/v3.0/High/wan22RemixT2VI2V_i2vHighV30-Q4_K_M.gguf",
    ModelVariant.WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: "High/wan22RemixT2VI2V_i2vHighV21-Q4_K_M.gguf",
    ModelVariant.WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: "Low/wan22RemixT2VI2V_i2vLowV21-Q4_K_M.gguf",
}

_IS_I2V = {
    ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: False,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: True,
    ModelVariant.WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: True,
    ModelVariant.WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: True,
}


class ModelLoader(ForgeModel):
    """Wan 2.2 Remix GGUF model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M: ModelConfig(
            pretrained_model_name=BIGDANNYPT_GGUF_REPO,
        ),
        ModelVariant.WAN22_REMIX_I2V_HIGH_V3_Q4_K_M: ModelConfig(
            pretrained_model_name=BIGDANNYPT_GGUF_REPO,
        ),
        ModelVariant.WAN22_REMIX_I2V_HIGH_V2_1_Q4_K_M: ModelConfig(
            pretrained_model_name=HUCHUKATO_I2V_V2_1_GGUF_REPO,
        ),
        ModelVariant.WAN22_REMIX_I2V_LOW_V2_1_Q4_K_M: ModelConfig(
            pretrained_model_name=HUCHUKATO_I2V_V2_1_GGUF_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_REMIX_T2V_HIGH_V2_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_REMIX_GGUF",
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
        """Load the GGUF-quantized Wan 2.2 Remix transformer and build the pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized transformer,
        then constructs the appropriate pipeline (T2V or I2V) with the base model's
        VAE in float32 for numerical stability.
        """
        from diffusers import (
            AutoencoderKLWan,
            GGUFQuantizationConfig,
            WanTransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        from huggingface_hub import hf_hub_download

        # gguf is installed at test time by RequirementsManager, but diffusers
        # caches availability flags and skips module-level imports when gguf is
        # absent. Patch the cached state and inject any symbols that were
        # conditionally skipped so the GGUF quantizer works at runtime.
        try:
            import importlib.metadata

            import gguf  # noqa: F401
            import diffusers.quantizers.gguf.gguf_quantizer as _gguf_mod
            import diffusers.utils.import_utils as _diu

            _diu._gguf_available = True
            _diu._gguf_version = importlib.metadata.version("gguf")

            if not hasattr(_gguf_mod, "_replace_with_gguf_linear"):
                from diffusers.quantizers.gguf.utils import (
                    GGML_QUANT_SIZES,
                    GGUFParameter,
                    _dequantize_gguf_and_restore_linear,
                    _quant_shape_from_byte_shape,
                    _replace_with_gguf_linear,
                )

                _gguf_mod.GGML_QUANT_SIZES = GGML_QUANT_SIZES
                _gguf_mod.GGUFParameter = GGUFParameter
                _gguf_mod._dequantize_gguf_and_restore_linear = (
                    _dequantize_gguf_and_restore_linear
                )
                _gguf_mod._quant_shape_from_byte_shape = _quant_shape_from_byte_shape
                _gguf_mod._replace_with_gguf_linear = _replace_with_gguf_linear
        except ImportError:
            pass

        is_i2v = _IS_I2V[self._variant]
        base_pipeline = I2V_BASE_PIPELINE if is_i2v else T2V_BASE_PIPELINE

        gguf_repo = _GGUF_REPOS[self._variant]
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        gguf_local_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_file)

        # _replace_with_gguf_linear always creates GGUFLinear on meta device. Any
        # model params absent from the GGUF file (e.g. biases) remain on meta after
        # load_model_dict_into_meta, causing dispatch_model to fail with
        # "Cannot copy out of meta tensor". Patch the symbol inside single_file_model
        # to materialize those stragglers before dispatch proceeds.
        #
        # Pass config= explicitly: without it, fetch_diffusers_config() maps any Wan
        # I2V GGUF to Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (which has image_dim != None
        # and thus add_k_proj), causing a dtype mismatch at runtime because those
        # params are float32 while activations are bfloat16.
        import diffusers.loaders.single_file_model as _sfm

        _orig_dispatch = _sfm.dispatch_model

        def _meta_safe_dispatch(model, device_map=None, **kw):
            if device_map:
                target = torch.device(list(device_map.values())[0])
                for pname, p in list(model.named_parameters(recurse=True)):
                    if p.device.type == "meta":
                        parts = pname.rsplit(".", 1)
                        parent = model
                        if len(parts) > 1:
                            for attr in parts[0].split("."):
                                parent = getattr(parent, attr)
                        parent._parameters[parts[-1]] = torch.nn.Parameter(
                            torch.empty(p.shape, dtype=p.dtype, device=target),
                            requires_grad=p.requires_grad,
                        )
                for bname, b in list(model.named_buffers(recurse=True)):
                    if b.device.type == "meta":
                        parts = bname.rsplit(".", 1)
                        parent = model
                        if len(parts) > 1:
                            for attr in parts[0].split("."):
                                parent = getattr(parent, attr)
                        parent._buffers[parts[-1]] = torch.empty(
                            b.shape, dtype=b.dtype, device=target
                        )
            return _orig_dispatch(model, device_map=device_map, **kw)

        _sfm.dispatch_model = _meta_safe_dispatch
        try:
            transformer = WanTransformer3DModel.from_single_file(
                gguf_local_path,
                config=base_pipeline,
                subfolder="transformer",
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
        finally:
            _sfm.dispatch_model = _orig_dispatch

        vae = AutoencoderKLWan.from_pretrained(
            base_pipeline,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        if is_i2v:
            from diffusers import WanImageToVideoPipeline

            # Wan 2.2 I2V does not use a CLIP image encoder (model_index.json has
            # image_encoder=[null,null]).  The pipeline also expects transformer_2
            # (low-noise expert); reuse the same GGUF transformer to avoid
            # downloading 28 GB of unquantized weights.
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                base_pipeline,
                transformer=transformer,
                transformer_2=transformer,
                vae=vae,
                torch_dtype=compute_dtype,
            )
        else:
            from diffusers import WanPipeline

            self.pipeline = WanPipeline.from_pretrained(
                base_pipeline,
                transformer=transformer,
                vae=vae,
                torch_dtype=compute_dtype,
            )

        return self.pipeline.transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Return synthetic inputs for the WanTransformer3DModel forward pass.

        Shapes derived from 480x832 @ 9 frames through the Wan VAE
        (spatial factor 8, temporal factor 4):
          latent: (1, in_channels, 3, 60, 104)
          text:   (1, 226, 4096)  [UMT5-XXL, max_seq_len=226]

        I2V in_channels=36 = 16 (noise) + 16 (reference latent) + 4 (mask).
        T2V in_channels=16.
        """
        if self.pipeline is None:
            self.load_model()

        is_i2v = _IS_I2V[self._variant]
        in_channels = 36 if is_i2v else 16
        dtype = torch.bfloat16

        height, width, num_frames = 480, 832, 9
        latent_h = height // 8
        latent_w = width // 8
        latent_t = (num_frames - 1) // 4 + 1

        return {
            "hidden_states": torch.randn(
                1, in_channels, latent_t, latent_h, latent_w, dtype=dtype
            ),
            "timestep": torch.tensor([500], dtype=torch.long),
            "encoder_hidden_states": torch.randn(1, 226, 4096, dtype=dtype),
            "return_dict": False,
        }
