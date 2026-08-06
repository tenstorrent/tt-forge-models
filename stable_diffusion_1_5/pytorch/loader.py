# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 1.5 model loader implementation.

SD1.4 (``CompVis/stable-diffusion-v1-4``) and SD1.5
(``stable-diffusion-v1-5/stable-diffusion-v1-5``) share the same
``UNet2DConditionModel`` architecture, ``LMSDiscreteScheduler`` and CLIP
text encoder; they only differ in pretrained weights. We still ship them
as separate loader packages so each bringup can advance independently and
each can carry its own ModelInfo / dashboards / status in the test runner.

``load_model`` returns the SD1.5 UNet (an ``nn.Module``) — the format the
tt-xla model tester expects.
"""

from typing import Optional

import torch
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

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
    """Available Stable Diffusion 1.5 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion 1.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with the requested variant.

        Args:
            variant: Optional ``ModelVariant``; falls back to ``DEFAULT_VARIANT``.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Stable Diffusion 1.5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SD1.5 UNet.

        Args:
            dtype_override: Optional ``torch.dtype`` for the UNet weights;
                defaults to ``torch.bfloat16`` to match TT execution.

        Returns:
            torch.nn.Module: The ``UNet2DConditionModel`` instance for SD1.5.
        """
        dtype = dtype_override or torch.bfloat16

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", **kwargs
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", **kwargs
        )
        unet = UNet2DConditionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=dtype,
            **kwargs,
        )
        self.scheduler = LMSDiscreteScheduler.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="scheduler",
            **kwargs,
        )

        self.in_channels = unet.in_channels
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return a single-step UNet sample input batch for SD1.5.

        Args:
            dtype_override: Optional ``torch.dtype``; defaults to ``torch.bfloat16``.
            batch_size: Repetition factor for the prompt.

        Returns:
            dict: ``{"sample": …, "timestep": 0, "encoder_hidden_states": …}``.
        """
        dtype = dtype_override or torch.bfloat16

        prompt = ["A fantasy landscape with mountains and rivers"] * batch_size
        text_input = self.tokenizer(prompt, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        height, width = 512, 512
        latents = torch.randn((batch_size, self.in_channels, height // 8, width // 8))

        num_inference_steps = 1
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        latent_model_input = self.scheduler.scale_model_input(latents, 0)

        return {
            "sample": latent_model_input.to(dtype),
            "timestep": 0,
            "encoder_hidden_states": text_embeddings.to(dtype),
        }

    # ------------------------------------------------------------------ #
    # TAESD lightweight VAE decoder on TT.
    #
    # SD1.5's full AutoencoderKL VAE noises out on TT. TAESD (madebyollin/taesd)
    # is the tiny AE for SD1.5's 4-channel latents; its conv-only decoder (no
    # complex/FFT, no GroupNorm) is the tractable decoder that runs on TT. See
    # tt-xla #5537.
    # ------------------------------------------------------------------ #

    TAESD_REPO = "madebyollin/taesd"

    def load_taesd_decoder(self):
        """Load TAESD, the tiny AE (tractable VAE decoder) for SD1.5 latents."""
        from diffusers import AutoencoderTiny

        self.taesd = (
            AutoencoderTiny.from_pretrained(self.TAESD_REPO, torch_dtype=torch.float32)
            .eval()
        )
        return self.taesd

    def decode_taesd(self, latents, on_tt=False):
        """Decode SD1.5 (4-ch) latents [B, 4, H/8, W/8] -> image [-1, 1] via TAESD.

        With ``on_tt=True`` the conv decoder runs on TT via
        ``torch.compile(backend="tt")``. ``AutoencoderTiny.decode(z)`` is
        ``self.decoder(z)`` directly (no pre-scaling), so raw latents feed the
        decoder in both modes. ``load_taesd_decoder`` must run first.
        """
        vae = getattr(self, "taesd", None) or self.load_taesd_decoder()
        with torch.no_grad():
            if not on_tt:
                return vae.decode(latents).sample

            import torch_xla  # noqa: F401
            import torch_xla.core.xla_model as xm

            dev = xm.xla_device()
            dec = vae.decoder.to(dtype=torch.bfloat16).to(dev)
            compiled = torch.compile(lambda z: dec(z), backend="tt")
            out = compiled(latents.to(dtype=torch.bfloat16).to(dev))
            return out.to("cpu").to(torch.float32)
