# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Fixer model loader implementation.

Fixer is a single-step image diffusion model (V2 of nvidia/difix) that
enhances rendered novel views by removing artifacts from 3D reconstructions
(NeRF/3DGS). It uses a Linear-attention Diffusion Transformer backbone with
a Deep Compression Autoencoder (DC-AE), built on top of Cosmos-Predict-0.6B.

Reference: https://huggingface.co/nvidia/Fixer
Upstream:  https://github.com/nv-tlabs/Fixer

Available variants:
- BASE: nvidia/Fixer (576x1024 image-to-image enhancement)
"""

import io
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
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

# Input image dimensions expected by the model
IMAGE_HEIGHT = 576
IMAGE_WIDTH = 1024

# HuggingFace repository and file paths
HF_REPO = "nvidia/Fixer"
CHECKPOINT_FILENAME = "pretrained/pretrained_fixer.pkl"
BASE_DIT_FILENAME = "base/model_fast_tokenizer.pt"
BASE_TOKENIZER_FILENAME = "base/tokenizer_fast.pth"

# Path to the cloned Fixer source repository
_FIXER_SRC = os.path.join(os.path.dirname(__file__), "_fixer_src")


class ModelVariant(StrEnum):
    """Available Fixer model variants."""

    BASE = "Base"


def _ensure_fixer_src():
    """Clone the Fixer repository if not already present."""
    if not os.path.isdir(_FIXER_SRC):
        import subprocess

        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/nv-tlabs/Fixer",
                _FIXER_SRC,
            ],
            capture_output=True,
            text=True,
        )
        src_dir = os.path.join(_FIXER_SRC, "src")
        if result.returncode != 0 and not os.path.isdir(src_dir):
            raise RuntimeError(f"Failed to clone Fixer repository: {result.stderr}")


def _patch_transformers():
    """Add SlidingWindowCache stub for transformers 5.x compatibility."""
    import transformers.cache_utils as cu

    if not hasattr(cu, "SlidingWindowCache"):

        class SlidingWindowCache(cu.DynamicCache):
            pass

        cu.SlidingWindowCache = SlidingWindowCache


def _patch_cosmos_tokenizer():
    """Patch CosmosImageTokenizer to load weights on CPU without CUDA."""
    try:
        import cosmos_predict2.tokenizers.tokenizer as _tok_mod

        cls = _tok_mod.CosmosImageTokenizer

        # Patch __init__ to use map_location="cpu" for torch.load and torch.jit.load
        _original_init = cls.__init__

        def _patched_init(self, *args, **kwargs):
            # Intercept the vae_pth to patch the load call
            vae_pth = kwargs.get("vae_pth") or (args[0] if args else None)
            if vae_pth is None:
                # Try to get it from the signature
                import inspect

                sig = inspect.signature(_original_init)
                params = list(sig.parameters.keys())
                if "vae_pth" in params:
                    idx = params.index("vae_pth") - 1  # -1 for self
                    if idx < len(args):
                        vae_pth = args[idx]

            # Call original init but with patched torch.load
            _original_torch_load = torch.load
            _original_jit_load = torch.jit.load

            def _cpu_torch_load(f, *a, **kw):
                kw["map_location"] = "cpu"
                return _original_torch_load(f, *a, **kw)

            def _cpu_jit_load(f, *a, **kw):
                kw["map_location"] = torch.device("cpu")
                return _original_jit_load(f, *a, **kw)

            torch.load = _cpu_torch_load
            torch.jit.load = _cpu_jit_load
            try:
                _original_init(self, *args, **kwargs)
            finally:
                torch.load = _original_torch_load
                torch.jit.load = _original_jit_load

        cls.__init__ = _patched_init

        # Patch register_mean_std to use CPU instead of cuda.current_device()
        _original_register = cls.register_mean_std

        def _patched_register_mean_std(self, mean, std):
            latent_mean, latent_std = mean, std
            latent_mean = latent_mean.to("cpu")
            latent_std = latent_std.to("cpu")
            latent_mean = latent_mean.view(self.latent_ch, -1)
            latent_std = latent_std.view(self.latent_ch, -1)
            target_shape = [1, self.latent_ch, 1, 1, 1]
            latent_mean = latent_mean.reshape(*target_shape)
            latent_std = latent_std.reshape(*target_shape)
            self.register_buffer(
                "latent_mean",
                latent_mean.to(self.dtype),
                persistent=False,
            )
            self.register_buffer(
                "latent_std",
                latent_std.to(self.dtype),
                persistent=False,
            )

        cls.register_mean_std = _patched_register_mean_std

    except ImportError:
        pass


def _patch_cosmos_pipeline():
    """Patch Text2ImagePipeline.from_config to use the device parameter for tensor_kwargs."""
    try:
        import cosmos_predict2.pipelines.text2image as _pipe_mod
        import types

        # Wrap from_config to fix tensor_kwargs after creation
        _original_from_config = _pipe_mod.Text2ImagePipeline.from_config.__func__

        @classmethod
        def _patched_from_config(
            cls,
            config,
            dit_path="",
            use_text_encoder=True,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
        ):
            pipe = _original_from_config(
                cls,
                config,
                dit_path=dit_path,
                use_text_encoder=use_text_encoder,
                device=device,
                torch_dtype=torch_dtype,
                load_ema_to_reg=load_ema_to_reg,
            )
            # Fix tensor_kwargs to use the requested device (not hardcoded "cuda")
            pipe.tensor_kwargs = {"device": device, "dtype": torch_dtype}
            return pipe

        _pipe_mod.Text2ImagePipeline.from_config = _patched_from_config
    except (ImportError, AttributeError):
        pass


def _apply_all_patches():
    """Apply all necessary patches for non-CUDA operation."""
    _patch_transformers()
    _patch_cosmos_tokenizer()
    _patch_cosmos_pipeline()


def _build_fixer_model(device="cpu", dtype=torch.bfloat16):
    """Build the Fixer model from cosmos-predict2 and load weights."""
    _ensure_fixer_src()

    fixer_src = os.path.join(_FIXER_SRC, "src")
    # Force fixer_src to position 0: other packages (e.g. torchxrayvision) may
    # insert their own dirs at sys.path[0] after we do, shadowing fixer's model.py.
    if fixer_src in sys.path:
        sys.path.remove(fixer_src)
    sys.path.insert(0, fixer_src)
    # Flush any previously-cached wrong 'model' module from sys.modules.
    sys.modules.pop("model", None)
    sys.modules.pop("pix2pix_turbo_nocond_cosmos_base_faster_tokenizer", None)

    orig_dir = os.getcwd()
    try:
        os.chdir(fixer_src)

        # Apply patches before importing cosmos_predict2 internals
        _apply_all_patches()

        from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import (  # noqa: PLC0415
            config,
        )
        from cosmos_predict2.pipelines.text2image import (  # noqa: PLC0415
            Text2ImagePipeline,
        )

        # Download base weights from HuggingFace
        dit_path = hf_hub_download(repo_id=HF_REPO, filename=BASE_DIT_FILENAME)
        tokenizer_path = hf_hub_download(
            repo_id=HF_REPO, filename=BASE_TOKENIZER_FILENAME
        )

        # Update config to use downloaded paths
        config.dit_path = dit_path
        config.tokenizer["vae_pth"] = tokenizer_path
        config.guardrail_config.enabled = False

        # Build the Cosmos model architecture on CPU
        pipeline = Text2ImagePipeline.from_config(
            config,
            dit_path=dit_path,
            use_text_encoder=False,
            device=device,
            torch_dtype=dtype,
        )

        # Load Fixer-specific fine-tuned weights
        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=CHECKPOINT_FILENAME)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pipeline.dit.load_state_dict(sd["state_dict_unet"], strict=False)
        pipeline.tokenizer.load_state_dict(sd["state_dict_vae"], strict=False)

        return pipeline
    finally:
        os.chdir(orig_dir)


class FixerWrapper(nn.Module):
    """Wraps the Fixer cosmos pipeline as a standard nn.Module.

    Single-step image enhancement: encode → denoise → decode.
    Input:  (B, 3, H, W) RGB image tensor
    Output: (B, 3, H, W) enhanced RGB image tensor
    """

    def __init__(self, pipeline, timestep=400):
        super().__init__()
        self.pipeline = pipeline

        # Precompute unconditional conditioning (Fixer runs unconditionally)
        from cosmos_predict2.conditioner import DataType  # noqa: PLC0415

        # Use model dtype for all condition tensors: torch.cat promotes mixed dtypes
        # to float32, which causes dtype mismatches when the DiT weights are bfloat16.
        dtype = next(pipeline.dit.parameters()).dtype
        batch_size = 1
        data_batch = {
            "dataset_name": "image_data",
            "images": torch.zeros(
                batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=dtype
            ),
            "t5_text_embeddings": torch.zeros(batch_size, 512, 1024, dtype=dtype),
            "fps": torch.ones((batch_size,), dtype=dtype) * 24,
            "padding_mask": torch.zeros(
                batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=dtype
            ),
        }
        _, uncondition = pipeline.conditioner.get_condition_uncondition(data_batch)
        self.condition = uncondition.edit_data_type(DataType.IMAGE)

        # Single diffusion timestep (sigma) for one-step inference
        self.register_buffer("sigma", torch.tensor([float(timestep)], dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            (B, 3, H, W) enhanced image tensor
        """
        dtype = next(self.pipeline.dit.parameters()).dtype

        # Expand to video format (B, C, T=1, H, W)
        x_5d = x.to(dtype).unsqueeze(2)

        # Encode with VAE; tokenizer may output float32 — cast to model dtype
        latent = self.pipeline.tokenizer.encode(x_5d).to(dtype)

        # Add noise at the specified sigma (single-step diffusion)
        sigma = self.sigma.to(dtype).expand(x.shape[0])
        noise = torch.randn_like(latent)
        xt = latent + sigma.view(-1, 1, 1, 1, 1) * noise

        # Denoise with DiT
        sigma_b_t = sigma.unsqueeze(1)  # (B, 1)
        prediction = self.pipeline.denoise(
            xt_B_C_T_H_W=xt,
            sigma=sigma_b_t,
            condition=self.condition,
        )
        z_denoised = prediction.x0

        # Decode with VAE and remove temporal dimension
        output = self.pipeline.tokenizer.decode(z_denoised)
        return output.squeeze(2)


class ModelLoader(ForgeModel):
    """NVIDIA Fixer model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="nvidia/Fixer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Fixer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fixer diffusion model."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        pipeline = _build_fixer_model(device="cpu", dtype=dtype)
        model = FixerWrapper(pipeline)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the Fixer model.

        Returns:
            torch.Tensor: RGB image tensor of shape (B, 3, 576, 1024).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        return torch.rand(
            batch_size,
            3,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            dtype=dtype,
        )
