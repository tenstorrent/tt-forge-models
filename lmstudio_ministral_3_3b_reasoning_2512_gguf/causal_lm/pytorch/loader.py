# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
lmstudio-community/Ministral-3-3B-Reasoning-2512-GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_gguf_mistral3():
    """Register mistral3 GGUF architecture with transformers.

    transformers 5.2.0 lacks a mapping for the mistral3 GGUF architecture.
    Ministral-3B uses the same GGUF config parameter layout as mistral, so we
    reuse that mapping and remap model_type from mistral3 to mistral.
    """
    import transformers.configuration_utils as config_utils_mod
    import transformers.modeling_gguf_pytorch_utils as gguf_utils_mod

    if "mistral3" in gguf_utils_mod.GGUF_SUPPORTED_ARCHITECTURES:
        return

    mistral_map = gguf_utils_mod.GGUF_TO_TRANSFORMERS_MAPPING["config"].get(
        "mistral", {}
    )
    gguf_utils_mod.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "mistral3"
    ] = mistral_map.copy()
    gguf_utils_mod.GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")

    _original_load = gguf_utils_mod.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kwargs
    ):
        result = _original_load(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
            **kwargs,
        )
        for section in ("config", "tokenizer_config"):
            if result.get(section, {}).get("model_type") == "mistral3":
                result[section]["model_type"] = "mistral"
        return result

    gguf_utils_mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    config_utils_mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    import transformers.models.auto.tokenization_auto as tokenization_auto_mod

    tokenization_auto_mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "mistral3" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["mistral3"] = GGUF_TO_FAST_CONVERTERS["llama"]


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
    """Available Ministral-3-3B-Reasoning-2512 GGUF model variants for causal language modeling."""

    MINISTRAL_3_3B_REASONING_2512_GGUF = "Ministral-3-3B-Reasoning-2512-GGUF"


class ModelLoader(ForgeModel):
    """lmstudio-community Ministral-3-3B-Reasoning-2512 GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_3B_REASONING_2512_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Ministral-3-3B-Reasoning-2512-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_3B_REASONING_2512_GGUF

    GGUF_FILE = "Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf"

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
            model="lmstudio Ministral-3-3B-Reasoning-2512 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_gguf_mistral3()
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
        _patch_gguf_mistral3()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
