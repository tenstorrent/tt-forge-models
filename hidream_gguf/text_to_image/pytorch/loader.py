# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HiDream GGUF model loader implementation for text-to-image generation.
"""

from typing import Any, Dict, Optional

import torch
from diffusers import DiffusionPipeline
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

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


class ModelVariant(StrEnum):
    """Available HiDream GGUF model variants."""

    HIDREAM_I1_FULL = "HiDream-I1-Full"


class ModelLoader(ForgeModel):
    """HiDream GGUF model loader implementation for text-to-image generation."""

    # text_encoder_4 and tokenizer_4 use Meta-Llama-3.1-8B-Instruct which is
    # stored in a separate gated HuggingFace repo and not bundled in the HiDream
    # repo. Use the ungated unsloth mirror with identical architecture.
    LLAMA_MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"

    _VARIANTS = {
        ModelVariant.HIDREAM_I1_FULL: ModelConfig(
            pretrained_model_name="HiDream-ai/HiDream-I1-Full",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HIDREAM_I1_FULL

    DEFAULT_PROMPT = "A beautiful sunset over a mountain landscape"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

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

        dtype = dtype_override if dtype_override is not None else torch.float32

        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            self.LLAMA_MODEL_ID, torch_dtype=dtype
        )
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(self.LLAMA_MODEL_ID)

        pipe_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "text_encoder_4": text_encoder_4,
            "tokenizer_4": tokenizer_4,
        }
        pipe_kwargs.update(extra_pipe_kwargs)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
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

    def load_inputs(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
