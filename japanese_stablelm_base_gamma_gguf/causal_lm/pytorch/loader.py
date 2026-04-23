# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Japanese StableLM Base Gamma 7B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
import inspect

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Available Japanese StableLM Base Gamma GGUF model variants."""

    JAPANESE_STABLELM_BASE_GAMMA_7B = "7B"


class ModelLoader(ForgeModel):
    """Japanese StableLM Base Gamma 7B GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.JAPANESE_STABLELM_BASE_GAMMA_7B: LLMModelConfig(
            pretrained_model_name="RichardErkhov/stabilityai_-_japanese-stablelm-base-gamma-7b-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JAPANESE_STABLELM_BASE_GAMMA_7B

    GGUF_FILE = "japanese-stablelm-base-gamma-7b.Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager."""
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    @staticmethod
    def _fix_gguf_model_to_load_compat():
        """Replace any monkey-patched load_gguf_checkpoint with the real transformers
        implementation so that model_to_load (required by transformers>=5) is preserved.

        Other GGUF loaders replace load_gguf_checkpoint with wrappers whose signatures
        pre-date the model_to_load kwarg. Loading a fresh copy of the module bypasses
        the entire patch chain and gives us the real function that accepts model_to_load.
        """
        import importlib.util

        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        import transformers.models.auto.tokenization_auto as tok_auto
        import transformers.configuration_utils as config_utils
        import transformers.modeling_utils as modeling_utils

        current_fn = gguf_utils.load_gguf_checkpoint
        sig = inspect.signature(current_fn)
        if "model_to_load" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            return

        # Load a fresh (unpatched) copy of the module to get the real function that
        # accepts model_to_load. The fresh module's functions use its own unpatched
        # get_gguf_hf_weights_map, which works for standard architectures like StableLM.
        spec = importlib.util.find_spec("transformers.modeling_gguf_pytorch_utils")
        fresh_spec = importlib.util.spec_from_file_location(
            "_tt_gguf_fresh", spec.origin
        )
        fresh_mod = importlib.util.module_from_spec(fresh_spec)
        fresh_spec.loader.exec_module(fresh_mod)
        real_fn = fresh_mod.load_gguf_checkpoint

        gguf_utils.load_gguf_checkpoint = real_fn
        for mod in (tok_auto, config_utils, modeling_utils):
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = real_fn

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
            model="Japanese StableLM Base Gamma GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
        self._fix_gguf_model_to_load_compat()
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
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
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
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
