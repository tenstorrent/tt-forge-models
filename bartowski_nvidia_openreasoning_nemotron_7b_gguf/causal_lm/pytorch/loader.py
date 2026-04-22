# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski nvidia OpenReasoning-Nemotron-7B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
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
    """Available bartowski nvidia OpenReasoning-Nemotron-7B GGUF model variants for causal language modeling."""

    BARTOWSKI_NVIDIA_OPENREASONING_NEMOTRON_7B_GGUF = "OpenReasoning_Nemotron_7B_GGUF"


class ModelLoader(ForgeModel):
    """bartowski nvidia OpenReasoning-Nemotron-7B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_NVIDIA_OPENREASONING_NEMOTRON_7B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/nvidia_OpenReasoning-Nemotron-7B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_NVIDIA_OPENREASONING_NEMOTRON_7B_GGUF

    GGUF_FILE = "nvidia_OpenReasoning-Nemotron-7B-Q4_K_M.gguf"

    sample_text = "Solve the equation 2x + 5 = 13 and explain your reasoning."

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
            model="bartowski nvidia OpenReasoning-Nemotron-7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _prepare_gguf_env():
        """Refresh stale package metadata and bypass load_gguf_checkpoint patch chain.

        RequirementsManager installs gguf at runtime, after transformers has
        already cached importlib.metadata.packages_distributions().  This
        causes is_gguf_available() to return version string 'N/A' and raises
        InvalidVersion.  Refreshing the mapping fixes that.

        Other GGUF loaders imported during collection wrap load_gguf_checkpoint
        in a chain that drops the model_to_load kwarg added in transformers 5.x.
        We traverse the closure chain to find the real transformers function and
        install a bypass wrapper that forwards all arguments including
        model_to_load.
        """
        import transformers.configuration_utils as _config_utils
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils
        import transformers.models.auto.tokenization_auto as _auto_tokenizer
        import transformers.tokenization_utils_tokenizers as _tok_utils
        import transformers.utils.import_utils as _import_utils

        _import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )

        _CHAIN_NAMES = {
            "_orig_load_gguf_checkpoint",
            "orig_load",
            "_inner",
            "_fn",
            "_real",
        }

        def _find_real_fn(fn):
            """Walk the patch chain to find the original transformers function.

            Each wrapper stores the previous function in a module global
            (e.g. _orig_load_gguf_checkpoint) rather than a closure, so we
            inspect __globals__ as well as __closure__ and __defaults__.
            """
            visited = set()
            queue = [fn]
            while queue:
                cur = queue.pop(0)
                if id(cur) in visited or not callable(cur):
                    continue
                visited.add(id(cur))
                code = getattr(cur, "__code__", None)
                if code and "modeling_gguf_pytorch_utils" in code.co_filename:
                    return cur
                globs = getattr(cur, "__globals__", {})
                for name in _CHAIN_NAMES:
                    val = globs.get(name)
                    if callable(val):
                        queue.append(val)
                for cell in getattr(cur, "__closure__", None) or ():
                    try:
                        queue.append(cell.cell_contents)
                    except ValueError:
                        pass
                for val in getattr(cur, "__defaults__", None) or ():
                    if callable(val):
                        queue.append(val)
            return None

        real_fn = _find_real_fn(_gguf_utils.load_gguf_checkpoint)
        if real_fn is None:
            return

        for mod in [_gguf_utils, _config_utils, _auto_tokenizer, _tok_utils]:
            if getattr(mod.load_gguf_checkpoint, "_nemotron_bypass", False):
                continue

            def _bypass(*args, _real=real_fn, **kwargs):
                return _real(*args, **kwargs)

            _bypass._nemotron_bypass = True
            mod.load_gguf_checkpoint = _bypass

    def _load_tokenizer(self, dtype_override=None):
        self._prepare_gguf_env()

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
        self._prepare_gguf_env()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
