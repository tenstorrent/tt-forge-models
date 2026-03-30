# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OFT ONNX model loader.
"""

import torch
import peft.tuners.oft.layer as oft_layer
from diffusers import StableDiffusionPipeline
from peft import OFTConfig, OFTModel

from ...base import ForgeModel
from ...config import Framework, ModelGroup, ModelInfo, ModelSource, ModelTask
from ...tools.utils import export_torch_model_to_onnx


class StableDiffusionWrapper(torch.nn.Module):
    """Wrap Stable Diffusion UNet forward for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latents, timesteps, prompt_embeds):
        return self.model.unet(
            latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample


class ModelLoader(ForgeModel):
    """OFT ONNX loader implementation."""

    DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = variant or self.DEFAULT_MODEL_NAME
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting."""
        if variant_name is None:
            variant_name = cls.DEFAULT_MODEL_NAME.split("/")[-1]
        return ModelInfo(
            model="OFT",
            variant=variant_name,
            group=ModelGroup.RED,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    @staticmethod
    def _get_oft_configs():
        config_te = OFTConfig(
            r=8,
            target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
            module_dropout=0.0,
            init_weights=True,
        )
        config_unet = OFTConfig(
            r=8,
            target_modules=[
                "proj_in",
                "proj_out",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            module_dropout=0.0,
            init_weights=True,
        )
        return config_te, config_unet

    @staticmethod
    def _patch_oft_cayley_with_lstsq():
        def _safe_cayley(self, data):
            data = data.detach()
            b, r, c = data.shape
            skew_mat = 0.5 * (data - data.transpose(1, 2))
            id_mat = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)
            try:
                return torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat)
            except RuntimeError:
                return torch.linalg.lstsq(id_mat + skew_mat, id_mat - skew_mat).solution

        oft_layer.Linear._cayley_batch = _safe_cayley

    @staticmethod
    def _freeze_oft_params(module):
        for _, layer in module.named_modules():
            if hasattr(layer, "oft_r"):
                for key in layer.oft_r:
                    layer.oft_r[key].requires_grad = False
            if hasattr(layer, "oft_s"):
                for key in layer.oft_s:
                    layer.oft_s[key].requires_grad = False

    def _build_pipeline(self):
        self._patch_oft_cayley_with_lstsq()
        pipe = StableDiffusionPipeline.from_pretrained(self.model_name)
        config_te, config_unet = self._get_oft_configs()
        pipe.text_encoder = OFTModel(pipe.text_encoder, config_te, "default")
        pipe.unet = OFTModel(pipe.unet, config_unet, "default")
        self._freeze_oft_params(pipe.text_encoder)
        self._freeze_oft_params(pipe.unet)
        pipe.to("cpu")
        pipe.text_encoder.eval()
        pipe.unet.eval()
        self.pipe = pipe

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load OFT model, export to ONNX, then load and return the ONNX model."""
        if self.pipe is None:
            self._build_pipeline()

        inputs = self.load_inputs(**kwargs)
        torch_model = StableDiffusionWrapper(self.pipe)
        model_name = f"{self.model_name.split('/')[-1].replace('-', '_')}_oft"

        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
            input_names=["latents", "timesteps", "prompt_embeds"],
            output_names=["sample"],
            do_constant_folding=True,
        )

    def load_inputs(
        self,
        prompt: str = "A beautiful mountain landscape during sunset",
        num_inference_steps: int = 30,
    ):
        """Load and return sample inputs for OFT ONNX export and compile."""
        if self.pipe is None:
            self._build_pipeline()

        prompt_embeds, negative_prompt_embeds, *_ = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            device="cpu",
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        height = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = height
        latents = torch.randn(
            (
                1,
                self.pipe.unet.config.in_channels,
                height // self.pipe.vae_scale_factor,
                width // self.pipe.vae_scale_factor,
            )
        )
        latents = latents * self.pipe.scheduler.init_noise_sigma
        latents = torch.cat([latents] * 2, dim=0)

        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timestep = self.pipe.scheduler.timesteps[0].expand(latents.shape[0])

        return (latents, timestep, prompt_embeds)
