# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash Uncensored Heretic NEO-CODE Imatrix MAX GGUF model loader
implementation for causal language modeling.
"""
import importlib.metadata
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available GLM-4.7-Flash Uncensored Heretic NEO-CODE Imatrix MAX GGUF variants."""

    GLM_4_7_FLASH_UNCENSORED_HERETIC_NEO_CODE_IMATRIX_MAX_GGUF = (
        "4.7_Flash_Uncensored_Heretic_NEO_CODE_Imatrix_MAX_GGUF"
    )


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash Uncensored Heretic NEO-CODE Imatrix MAX GGUF model loader
    implementation for causal language modeling tasks."""

    @staticmethod
    def _find_original_load_gguf_checkpoint():
        """Walk the monkey-patch chain to find the real transformers load_gguf_checkpoint."""
        import inspect

        import transformers.modeling_gguf_pytorch_utils as gguf_utils

        _ORIG_NAMES = (
            "_orig_load_gguf_checkpoint",
            "_orig",
            "orig_load",
            "_real",
            "_current_load",
            "_base_load",
        )

        fn = gguf_utils.load_gguf_checkpoint
        seen = set()
        while True:
            fn_id = id(fn)
            if fn_id in seen:
                break
            seen.add(fn_id)
            try:
                src = inspect.getfile(fn)
            except (TypeError, OSError):
                break
            if "tt_forge_models" not in src and "worktrees" not in src:
                return fn
            if hasattr(fn, "__wrapped__"):
                fn = fn.__wrapped__
                continue
            unwrapped = None
            if fn.__closure__ and fn.__code__.co_freevars:
                for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                    if name.startswith(
                        ("_orig", "orig_", "_real", "_current", "_base")
                    ):
                        try:
                            val = cell.cell_contents
                            if callable(val) and id(val) not in seen:
                                unwrapped = val
                                break
                        except ValueError:
                            pass
            if unwrapped is None and hasattr(fn, "__globals__"):
                for name in _ORIG_NAMES:
                    val = fn.__globals__.get(name)
                    if callable(val) and id(val) not in seen:
                        unwrapped = val
                        break
            if unwrapped is not None:
                fn = unwrapped
                continue
            break
        return fn

    @staticmethod
    def _fix_gguf_loading():
        """Fix gguf version detection, patch chain, and register missing tokenizer converters.

        1. Clears the lru_cache on is_gguf_available so runtime gguf install is detected.
        2. Replaces the monkey-patch chain with a clean wrapper of the original transformers
           load_gguf_checkpoint that accepts **kwargs (incl. model_to_load from transformers 5.x).
        3. Registers deepseek_v2/deepseek2 in GGUF_TO_FAST_CONVERTERS for the GLM tokenizer.
        """
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as _import_utils
        from transformers.integrations.ggml import (
            GGUF_TO_FAST_CONVERTERS,
            GGUFQwen2Converter,
        )

        try:
            importlib.metadata.version("gguf")
            if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
            _import_utils.is_gguf_available.cache_clear()
        except importlib.metadata.PackageNotFoundError:
            pass

        _orig = ModelLoader._find_original_load_gguf_checkpoint()

        def _patched(*args, **kwargs):
            return _orig(*args, **kwargs)

        _patched.__wrapped__ = _orig
        gguf_utils.load_gguf_checkpoint = _patched
        if hasattr(modeling_utils, "load_gguf_checkpoint"):
            modeling_utils.load_gguf_checkpoint = _patched

        for key in ("deepseek2", "deepseek_v2"):
            if key not in GGUF_TO_FAST_CONVERTERS:
                GGUF_TO_FAST_CONVERTERS[key] = GGUFQwen2Converter

        if not getattr(
            gguf_utils.get_gguf_hf_weights_map, "_deepseek_v2_patched", False
        ):
            _orig_get_map = gguf_utils.get_gguf_hf_weights_map

            def _patched_get_map(
                hf_model, processor, model_type=None, num_layers=None, qual_name=""
            ):
                if model_type is None and hasattr(hf_model, "config"):
                    model_type = hf_model.config.model_type
                if model_type == "deepseek_v2":
                    model_type = "deepseek2"
                return _orig_get_map(
                    hf_model, processor, model_type, num_layers, qual_name
                )

            _patched_get_map.__wrapped__ = _orig_get_map
            _patched_get_map._deepseek_v2_patched = True
            gguf_utils.get_gguf_hf_weights_map = _patched_get_map

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_UNCENSORED_HERETIC_NEO_CODE_IMATRIX_MAX_GGUF: LLMModelConfig(
            pretrained_model_name="Ihabb213/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.GLM_4_7_FLASH_UNCENSORED_HERETIC_NEO_CODE_IMATRIX_MAX_GGUF
    )

    GGUF_FILE = "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-Q4_K_M.gguf"

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
            model="GLM-4.7-Flash Uncensored Heretic NEO-CODE Imatrix MAX GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_loading()
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
        self._fix_gguf_loading()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["ignore_mismatched_sizes"] = True

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
        self._fix_gguf_loading()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
