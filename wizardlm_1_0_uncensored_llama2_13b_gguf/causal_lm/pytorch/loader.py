# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TheBloke WizardLM 1.0 Uncensored Llama2 13B GGUF model loader implementation for causal language modeling.
"""
import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.configuration_utils as _config_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils

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


def _find_real_load_gguf_checkpoint():
    """Find the real transformers load_gguf_checkpoint by scanning sys.modules globals.

    Other model loaders patch _gguf_utils.load_gguf_checkpoint with restrictive signatures
    that drop model_to_load (required by transformers 5.x). The real original is identifiable
    by having both gguf_checkpoint_path and model_to_load in its parameter list.
    """
    import sys

    for mod in list(sys.modules.values()):
        mod_dict = getattr(mod, "__dict__", None)
        if mod_dict is None:
            continue
        for obj in list(mod_dict.values()):
            if not callable(obj):
                continue
            try:
                params = inspect.signature(obj).parameters
                if "gguf_checkpoint_path" in params and "model_to_load" in params:
                    return obj
            except (ValueError, TypeError):
                pass
    return _gguf_utils.load_gguf_checkpoint


def _apply_gguf_patch():
    """Re-patch load_gguf_checkpoint to restore model_to_load support.

    Other loaders may install restrictive patches that drop model_to_load (required by
    transformers 5.x). This finds the real original and re-wraps it.
    """
    real_fn = _find_real_load_gguf_checkpoint()

    def _patched(*args, **kwargs):
        return real_fn(*args, **kwargs)

    _gguf_utils.load_gguf_checkpoint = _patched
    _config_utils.load_gguf_checkpoint = _patched
    _auto_tokenizer.load_gguf_checkpoint = _patched
    _tok_utils.load_gguf_checkpoint = _patched


class ModelVariant(StrEnum):
    """Available WizardLM 1.0 Uncensored Llama2 13B GGUF model variants for causal language modeling."""

    WIZARDLM_1_0_UNCENSORED_LLAMA2_13B_GGUF = "13B_GGUF"


class ModelLoader(ForgeModel):
    """TheBloke WizardLM 1.0 Uncensored Llama2 13B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.WIZARDLM_1_0_UNCENSORED_LLAMA2_13B_GGUF: LLMModelConfig(
            pretrained_model_name="TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WIZARDLM_1_0_UNCENSORED_LLAMA2_13B_GGUF

    GGUF_FILE = "wizardlm-1.0-uncensored-llama2-13b.Q4_K_M.gguf"

    sample_text = (
        "What are the key differences between classical and quantum computing?"
    )

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
            model="WizardLM 1.0 Uncensored Llama2 13B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _apply_gguf_patch()
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
        _apply_gguf_patch()
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
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
        _apply_gguf_patch()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
