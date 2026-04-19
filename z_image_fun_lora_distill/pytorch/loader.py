# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Fun-Lora-Distill model loader implementation.

Distill LoRA adapter for the Z-Image diffusion transformer that distills both
inference steps and classifier-free guidance (CFG) for fast text-to-image
generation. Trained from scratch (not based on Z-Image-Turbo weights).

Available variants:
- DISTILL_8_STEPS: alibaba-pai/Z-Image-Fun-Lora-Distill 8-step distilled LoRA
"""

from typing import Optional, Dict, Any, List

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors import safe_open

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


BASE_MODEL_ID = "Tongyi-MAI/Z-Image"


class ModelVariant(StrEnum):
    """Available Z-Image-Fun-Lora-Distill model variants."""

    DISTILL_8_STEPS = "Distill_8_Steps"


class ModelLoader(ForgeModel):
    """Z-Image-Fun-Lora-Distill model loader implementation."""

    _VARIANTS = {
        ModelVariant.DISTILL_8_STEPS: ModelConfig(
            pretrained_model_name="alibaba-pai/Z-Image-Fun-Lora-Distill",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DISTILL_8_STEPS

    DEFAULT_PROMPT = "A serene mountain landscape at sunrise, photorealistic, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z-Image-Fun-Lora-Distill",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
    ) -> None:
        pipe_dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=pipe_dtype,
        )

        # The LoRA file uses kohya key format with double underscores
        # (lora_unet__...) which triggers a diffusers conversion bug producing
        # "transformer.." (double-dot) keys. Normalize to single underscore.
        adapter_id = self._variant_config.pretrained_model_name
        weight_name = "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors"
        lora_path = hf_hub_download(adapter_id, weight_name)
        with safe_open(lora_path, framework="pt") as f:
            lora_sd = {
                k.replace("lora_unet__", "lora_unet_"): f.get_tensor(k)
                for k in f.keys()
            }
        self.pipeline.load_lora_weights(lora_sd)
        self.pipeline.fuse_lora(lora_scale=0.8)

        self.pipeline.to("cpu")

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(
                dtype=dtype_override
            )

        return self.pipeline.transformer

    def _encode_prompt(self, prompt: str) -> List[torch.Tensor]:
        pipe = self.pipeline
        messages = [{"role": "user", "content": prompt}]
        text = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = pipe.tokenizer(
            [text],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        mask = inputs.attention_mask.bool()
        hidden = pipe.text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]
        return [hidden[0][mask[0]]]

    def load_inputs(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        model_dtype = self.pipeline.transformer.dtype

        prompt_embeds = self._encode_prompt(prompt_value)
        prompt_embeds = [pe.to(model_dtype) for pe in prompt_embeds]

        height = width = 512
        vae_sf = self.pipeline.vae_scale_factor
        latent_h = 2 * (height // (vae_sf * 2))
        latent_w = 2 * (width // (vae_sf * 2))
        in_channels = self.pipeline.transformer.in_channels
        latents = torch.randn(1, in_channels, latent_h, latent_w, dtype=model_dtype)
        latents = latents.unsqueeze(2)
        x = list(latents.unbind(dim=0))

        t = torch.tensor([500.0], dtype=model_dtype)

        return {"x": x, "t": t, "cap_feats": prompt_embeds, "return_dict": False}
