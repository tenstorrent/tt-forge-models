# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternLM2.5 20B Chat IMat GGUF model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel


def _patch_internlm2_gguf_support():
    """Patch transformers to recognise the 'internlm2' GGUF architecture.

    Transformers 5.x has no GGUF loader for internlm2. The architecture
    uses the same GGUF key layout and tensor names as llama, so we register
    the mapping and remap model_type to 'llama' so that LlamaForCausalLM
    handles inference.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_utils as _modeling_utils
    from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "internlm2" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("internlm2")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "internlm2"
    ] = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["llama"].copy()

    if "llama" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "internlm2", GGUF_TO_FAST_CONVERTERS["llama"]
        )

    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        cfg = result.get("config", {})
        if cfg.get("model_type") == "internlm2":
            cfg["model_type"] = "llama"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load
    for _mod in (_config_utils, _modeling_utils):
        if hasattr(_mod, "load_gguf_checkpoint"):
            _mod.load_gguf_checkpoint = _patched_load

    try:
        import transformers.models.auto.tokenization_auto as _tok_auto

        if hasattr(_tok_auto, "load_gguf_checkpoint"):
            _tok_auto.load_gguf_checkpoint = _patched_load
    except ImportError:
        pass

    try:
        import transformers.tokenization_utils_tokenizers as _tok_utils

        if hasattr(_tok_utils, "load_gguf_checkpoint"):
            _tok_utils.load_gguf_checkpoint = _patched_load
    except ImportError:
        pass


_patch_internlm2_gguf_support()
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
    """Available InternLM2.5 20B Chat IMat GGUF model variants for causal language modeling."""

    INTERNLM2_5_20B_CHAT_Q4_K_IMAT_GGUF = "InternLM2_5_20B_Chat_Q4_K_IMat_GGUF"


class ModelLoader(ForgeModel):
    """InternLM2.5 20B Chat IMat GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.INTERNLM2_5_20B_CHAT_Q4_K_IMAT_GGUF: LLMModelConfig(
            pretrained_model_name="legraphista/internlm2_5-20b-chat-IMat-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERNLM2_5_20B_CHAT_Q4_K_IMAT_GGUF

    GGUF_FILE = "internlm2_5-20b-chat.Q4_K.gguf"

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
            model="InternLM2.5 20B Chat IMat GGUF",
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

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                gguf_file=self.GGUF_FILE,
                trust_remote_code=True,
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
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            trust_remote_code=True,
        )
        return self.config
