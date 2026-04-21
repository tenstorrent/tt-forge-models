# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HiDream GGUF model loader implementation for text-to-image generation.
"""

from typing import Any, Dict, Optional

import torch
from diffusers import (
    DiffusionPipeline,
    GGUFQuantizationConfig,
    HiDreamImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

CITY96_GGUF_REPO = "city96/HiDream-I1-Full-gguf"


class ModelVariant(StrEnum):
    """Available HiDream GGUF model variants."""

    HIDREAM_I1_FULL = "HiDream-I1-Full"
    HIDREAM_I1_FULL_Q4_K_S = "HiDream-I1-Full-Q4_K_S"
    HIDREAM_I1_FULL_Q8_0 = "HiDream-I1-Full-Q8_0"


class ModelLoader(ForgeModel):
    """HiDream GGUF model loader implementation for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.HIDREAM_I1_FULL: ModelConfig(
            pretrained_model_name="HiDream-ai/HiDream-I1-Full",
        ),
        ModelVariant.HIDREAM_I1_FULL_Q4_K_S: ModelConfig(
            pretrained_model_name=CITY96_GGUF_REPO,
        ),
        ModelVariant.HIDREAM_I1_FULL_Q8_0: ModelConfig(
            pretrained_model_name=CITY96_GGUF_REPO,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.HIDREAM_I1_FULL_Q4_K_S: "hidream-i1-full-Q4_K_S.gguf",
        ModelVariant.HIDREAM_I1_FULL_Q8_0: "hidream-i1-full-Q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.HIDREAM_I1_FULL

    DEFAULT_PROMPT = "A beautiful sunset over a mountain landscape"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None
        self.transformer: Optional[HiDreamImageTransformer2DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HiDream GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DiffusionPipeline:
        if extra_pipe_kwargs is None:
            extra_pipe_kwargs = {}

        pipe_kwargs = {
            "torch_dtype": (
                dtype_override if dtype_override is not None else torch.float32
            ),
            "device_map": device_map,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        pipe_kwargs.update(extra_pipe_kwargs)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def _load_gguf_transformer(
        self,
        dtype_override: Optional[torch.dtype] = None,
    ) -> HiDreamImageTransformer2DModel:
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self._GGUF_FILES[self._variant],
        )
        self.transformer = HiDreamImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
            torch_dtype=compute_dtype,
        )
        self.transformer.eval()
        return self.transformer

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if self._variant in self._GGUF_FILES:
            if self.transformer is None:
                return self._load_gguf_transformer(dtype_override=dtype_override)
            if dtype_override is not None:
                self.transformer = self.transformer.to(dtype=dtype_override)
            return self.transformer

        if self.pipeline is None:
            return self._load_pipeline(
                dtype_override=dtype_override,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                extra_pipe_kwargs=extra_pipe_kwargs,
            )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def _load_gguf_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        if self.transformer is None:
            self._load_gguf_transformer(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Latent dimensions (VAE scale factor 8, patch size from config).
        height = 128
        width = 128
        vae_scale_factor = 8
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor
        in_channels = config.in_channels

        hidden_states = torch.randn(
            batch_size, in_channels, latent_h, latent_w, dtype=dtype
        )
        timesteps = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # caption_channels is [pooled/T5, llama3]; see transformer_hidream_image.
        t5_channels, llama3_channels = config.caption_channels
        text_seq_len = 128
        encoder_hidden_states_t5 = torch.randn(
            batch_size, text_seq_len, t5_channels, dtype=dtype
        )
        num_llama_layers = len(config.llama_layers)
        encoder_hidden_states_llama3 = torch.randn(
            num_llama_layers, batch_size, text_seq_len, llama3_channels, dtype=dtype
        )
        pooled_embeds = torch.randn(batch_size, config.text_emb_dim, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timesteps": timesteps,
            "encoder_hidden_states_t5": encoder_hidden_states_t5,
            "encoder_hidden_states_llama3": encoder_hidden_states_llama3,
            "pooled_embeds": pooled_embeds,
        }

    def load_inputs(
        self,
        prompt: Optional[str] = None,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        if self._variant in self._GGUF_FILES:
            return self._load_gguf_inputs(
                dtype_override=dtype_override, batch_size=batch_size
            )
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
