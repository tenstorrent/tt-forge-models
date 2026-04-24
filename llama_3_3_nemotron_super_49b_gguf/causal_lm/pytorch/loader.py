# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.3 Nemotron Super 49B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
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


class DeciLlamaConfig(LlamaConfig):
    """LlamaConfig that accepts per-layer list values from the deci GGUF architecture.

    The deci/DeciLM architecture uses NAS-designed variable per-layer configs.
    This class normalises list-valued fields to a single scalar so the standard
    LlamaForCausalLM can be used for compile testing with random weights.
    """

    model_type = "deci"

    def __init__(self, **kwargs):
        for field in (
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
        ):
            value = kwargs.get(field)
            if isinstance(value, list):
                non_zero = [v for v in value if v > 0]
                kwargs[field] = (
                    max(set(non_zero), key=non_zero.count) if non_zero else value[0]
                )
        super().__init__(**kwargs)


# Register deci GGUF architecture so AutoConfig and AutoTokenizer can resolve it.
if "deci" not in CONFIG_MAPPING._extra_content:
    CONFIG_MAPPING.register("deci", DeciLlamaConfig)


class ModelVariant(StrEnum):
    """Available Llama 3.3 Nemotron Super 49B GGUF model variants for causal language modeling."""

    LLAMA_3_3_NEMOTRON_SUPER_49B_V1_GGUF = "3.3_Nemotron_Super_49B_v1_GGUF"


class ModelLoader(ForgeModel):
    """Llama 3.3 Nemotron Super 49B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_3_NEMOTRON_SUPER_49B_V1_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_3_NEMOTRON_SUPER_49B_V1_GGUF

    GGUF_FILE = "nvidia_Llama-3_3-Nemotron-Super-49B-v1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

    # Scalar config values derived from the deci GGUF metadata.  The deci
    # architecture stores per-layer lists for several fields; we use the modal
    # non-zero value so a uniform LlamaForCausalLM can be constructed.
    _LLAMA_CONFIG_KWARGS = dict(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        vocab_size=128256,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        bos_token_id=128000,
        eos_token_id=128009,
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
            model="Llama 3.3 Nemotron Super 49B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _build_config(self):
        kwargs = dict(self._LLAMA_CONFIG_KWARGS)
        if self.num_layers is not None:
            kwargs["num_hidden_layers"] = self.num_layers
        return DeciLlamaConfig(**kwargs)

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self._build_config()
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LlamaForCausalLM(config).eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = config
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
        self.config = self._build_config()
        return self.config
