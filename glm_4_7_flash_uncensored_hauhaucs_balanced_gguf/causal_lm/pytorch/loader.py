# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash-Uncensored-HauhauCS-Balanced GGUF model loader for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _patch_transformers_deepseek_v2_gguf():
    """Monkey-patch transformers to support deepseek_v2 in GGUF tokenizer and weight loading.

    The glm_4_7_flash_gguf loader remaps model_type from 'deepseek2' to
    'deepseek_v2' globally. This causes two failures for models using the
    deepseek2 GGUF architecture:
      1. convert_gguf_tokenizer looks up GGUF_TO_FAST_CONVERTERS['deepseek_v2']
      2. get_gguf_hf_weights_map looks up 'deepseek_v2' in MODEL_ARCH_NAMES

    Both are fixed by aliasing deepseek_v2 -> deepseek2 converters.
    """
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter

    if not getattr(gguf_utils, "_deepseek_v2_weights_map_patched", False):
        orig_get_map = gguf_utils.get_gguf_hf_weights_map

        def patched_get_gguf_hf_weights_map(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            if model_type is None:
                cfg = getattr(hf_model, "config", None)
                model_type = getattr(cfg, "model_type", None)
            if model_type == "deepseek_v2":
                model_type = "deepseek2"
            return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

        gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map
        gguf_utils._deepseek_v2_weights_map_patched = True


_patch_transformers_deepseek_v2_gguf()

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GLM-4.7-Flash-Uncensored-HauhauCS-Balanced GGUF model variants."""

    GLM_4_7_FLASH_UNCENSORED_HAUHAUCS_BALANCED_GGUF = (
        "4.7_Flash_Uncensored_HauhauCS_Balanced_GGUF"
    )


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash-Uncensored-HauhauCS-Balanced GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_UNCENSORED_HAUHAUCS_BALANCED_GGUF: LLMModelConfig(
            pretrained_model_name="HauhauCS/GLM-4.7-Flash-Uncensored-HauhauCS-Balanced",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_UNCENSORED_HAUHAUCS_BALANCED_GGUF

    GGUF_FILE = "GLM-4.7-Flash-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="GLM-4.7-Flash-Uncensored-HauhauCS-Balanced GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
