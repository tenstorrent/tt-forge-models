# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Small 4 119B 2603 GGUF model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _apply_mistral4_gguf_patch():
    """Register mistral4 GGUF architecture as an alias for mistral.

    Mistral Small 4 GGUF files declare architecture as 'mistral4' but transformers
    only recognises 'mistral'. The config field layout is identical to 'mistral'.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        load_gguf_checkpoint as _orig_load,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "mistral4" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("mistral4")

    if (
        "mistral4" not in GGUF_TO_TRANSFORMERS_MAPPING["config"]
        and "mistral" in GGUF_TO_TRANSFORMERS_MAPPING["config"]
    ):
        GGUF_TO_TRANSFORMERS_MAPPING["config"][
            "mistral4"
        ] = GGUF_TO_TRANSFORMERS_MAPPING["config"]["mistral"]

    for key in ("llama",):
        if key in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS.setdefault("mistral4", GGUF_TO_FAST_CONVERTERS[key])
            GGUF_TO_FAST_CONVERTERS.setdefault("mistral", GGUF_TO_FAST_CONVERTERS[key])
            break

    from transformers.modeling_gguf_pytorch_utils import (
        get_gguf_hf_weights_map as _orig_get_map,
    )

    def _patched_get_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        effective_type = (
            hf_model.config.model_type if model_type is None else model_type
        )
        if effective_type == "mistral":
            model_type = "llama"
        return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map

    def _patched_load(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kwargs
    ):
        import inspect

        sig = inspect.signature(_orig_load)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        result = _orig_load(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
            **valid_kwargs,
        )
        if result.get("config", {}).get("model_type") == "mistral4":
            result["config"]["model_type"] = "mistral"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load
    _config_utils.load_gguf_checkpoint = _patched_load
    _auto_tokenizer.load_gguf_checkpoint = _patched_load
    try:
        import transformers.tokenization_utils_tokenizers as _tok_utils

        _tok_utils.load_gguf_checkpoint = _patched_load
    except ImportError:
        pass


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
    """Available Mistral Small 4 119B 2603 GGUF model variants for causal language modeling."""

    MISTRAL_SMALL_4_119B_IQ4_XS_GGUF = "119B_IQ4_XS_GGUF"
    MRADERMACHER_MISTRAL_SMALL_4_119B_Q4_K_M_GGUF = "mradermacher_119B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mistral Small 4 119B 2603 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_4_119B_IQ4_XS_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/Mistral-Small-4-119B-2603-GGUF",
            max_length=128,
        ),
        ModelVariant.MRADERMACHER_MISTRAL_SMALL_4_119B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Mistral-Small-4-119B-2603-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_4_119B_IQ4_XS_GGUF

    _GGUF_FILES = {
        ModelVariant.MISTRAL_SMALL_4_119B_IQ4_XS_GGUF: "IQ4_XS/Mistral-Small-4-119B-2603-IQ4_XS-00001-of-00003.gguf",
        ModelVariant.MRADERMACHER_MISTRAL_SMALL_4_119B_Q4_K_M_GGUF: "Mistral-Small-4-119B-2603.Q4_K_M.gguf",
    }

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mistral Small 4 119B 2603 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _apply_mistral4_gguf_patch()

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _apply_mistral4_gguf_patch()

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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
        _apply_mistral4_gguf_patch()

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
