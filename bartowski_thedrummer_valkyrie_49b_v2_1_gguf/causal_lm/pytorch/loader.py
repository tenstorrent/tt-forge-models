# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Valkyrie 49B v2.1 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_deci_gguf_arch():
    """Patch transformers to map the 'deci' GGUF architecture to 'llama'.

    The Valkyrie 49B v2.1 GGUF uses general.architecture = 'deci' (DeciLM),
    which is a Llama-based architecture. Transformers 5.x has 'deci' in
    GGUF_CONFIG_MAPPING and GGUF_TO_FAST_CONVERTERS (GGUFLlamaConverter),
    but not in AutoConfig's CONFIG_MAPPING. AutoTokenizer.from_pretrained
    and AutoConfig.from_pretrained both call AutoConfig.for_model(model_type=
    'deci') which raises ValueError. The fix: patch load_gguf_checkpoint to
    remap model_type 'deci' -> 'llama' before it reaches AutoConfig.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if getattr(gguf_utils, "_deci_arch_patched", False):
        return

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "deci":
            config["model_type"] = "llama"
            # DeciLM stores per-layer values as lists; LlamaConfig expects scalars.
            # Use the max non-zero value as an approximation for each field.
            for key in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
                if isinstance(config.get(key), list):
                    non_zero = [v for v in config[key] if v != 0]
                    config[key] = max(non_zero) if non_zero else config[key][0]
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    gguf_utils._deci_arch_patched = True

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils

    for mod in (tok_auto, config_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_deci_gguf_arch()

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
    """Available Valkyrie 49B v2.1 GGUF model variants for causal language modeling."""

    VALKYRIE_49B_V2_1_GGUF = "49B_V2_1_GGUF"


class ModelLoader(ForgeModel):
    """Valkyrie 49B v2.1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.VALKYRIE_49B_V2_1_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/TheDrummer_Valkyrie-49B-v2.1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VALKYRIE_49B_V2_1_GGUF

    GGUF_FILE = "TheDrummer_Valkyrie-49B-v2.1-Q4_K_M.gguf"

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
            model="Valkyrie 49B v2.1 GGUF",
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
