# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron-Cascade model loader implementation for causal language modeling.
"""

import contextlib
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Nemotron-Cascade model variants for causal language modeling."""

    NEMOTRON_CASCADE_2_30B_A3B = "Cascade_2_30B_A3B"
    NEMOTRON_CASCADE_2_30B_A3B_MLX_6BIT = "Cascade_2_30B_A3B_mlx_6bit"


class ModelLoader(ForgeModel):
    """Nemotron-Cascade model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_CASCADE_2_30B_A3B: LLMModelConfig(
            pretrained_model_name="nvidia/Nemotron-Cascade-2-30B-A3B",
            max_length=128,
        ),
        ModelVariant.NEMOTRON_CASCADE_2_30B_A3B_MLX_6BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/Nemotron-Cascade-2-30B-A3B-6bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_CASCADE_2_30B_A3B

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nemotron-Cascade",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        if self._variant == ModelVariant.NEMOTRON_CASCADE_2_30B_A3B_MLX_6BIT:
            model_kwargs["device_map"] = "cpu"
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if not torch.cuda.is_available():
            self._patch_cuda_streams_for_cpu()

        model.eval()
        self.config = model.config

        return model

    @staticmethod
    def _patch_cuda_streams_for_cpu():
        """Patch NemotronHBlock.forward to use nullcontext instead of CUDA streams on CPU."""
        for mod_name, mod in sys.modules.items():
            if "modeling_nemotron_h" in mod_name and hasattr(mod, "NemotronHBlock"):
                cls = mod.NemotronHBlock

                def _cpu_forward(
                    self,
                    hidden_states,
                    cache_params=None,
                    cache_position=None,
                    attention_mask=None,
                ):
                    with contextlib.nullcontext():
                        residual = hidden_states
                        hidden_states = self.norm(
                            hidden_states.to(dtype=self.norm.weight.dtype)
                        )
                        if self.residual_in_fp32:
                            residual = residual.to(torch.float32)
                        if self.block_type == "mamba":
                            hidden_states = self.mixer(
                                hidden_states,
                                cache_params=cache_params,
                                cache_position=cache_position,
                            )
                        elif self.block_type == "attention":
                            hidden_states = self.mixer(
                                hidden_states, cache_position=cache_position
                            )
                            hidden_states = hidden_states[0]
                        elif self.block_type in ["mlp", "moe"]:
                            hidden_states = self.mixer(hidden_states)
                        else:
                            raise ValueError(f"Invalid block_type: {self.block_type}")
                        hidden_states = residual + hidden_states
                        return hidden_states

                cls.forward = _cpu_forward
                break

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
