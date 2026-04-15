# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DreamOmni2 GGUF model loader implementation for causal language modeling.

The GGUF repo (rafacost/DreamOmni2-7.6B-GGUF) uses the qwen2vl architecture
which is not yet supported by transformers' GGUF loader.  We therefore load
from the canonical repo (xiabs/DreamOmni2, subfolder vlm-model) which carries
the full config and safetensors weights.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoConfig
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
    """Available DreamOmni2 GGUF model variants for causal language modeling."""

    DREAM_OMNI_2_7_6B_Q4_K_M_GGUF = "Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """DreamOmni2 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DREAM_OMNI_2_7_6B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="rafacost/DreamOmni2-7.6B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DREAM_OMNI_2_7_6B_Q4_K_M_GGUF

    _CANONICAL_REPO = "xiabs/DreamOmni2"
    _SUBFOLDER = "vlm-model"

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DreamOmni2 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._CANONICAL_REPO, subfolder=self._SUBFOLDER
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Always pre-load config so the random-weights interceptor does not
        # need to resolve the subfolder itself.
        if "config" not in model_kwargs:
            config = AutoConfig.from_pretrained(
                self._CANONICAL_REPO, subfolder=self._SUBFOLDER
            )
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._CANONICAL_REPO, subfolder=self._SUBFOLDER, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._CANONICAL_REPO, subfolder=self._SUBFOLDER
        )
        return self.config
