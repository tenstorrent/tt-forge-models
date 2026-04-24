# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/Kai-3B-Distill-Claude-Opus-4.5-GGUF model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel


def _patch_transformers_smollm3_gguf():
    """Monkey-patch transformers to add smollm3 GGUF architecture support.

    Transformers 5.x has SmolLM3ForCausalLM but lacks GGUF loading support
    for the smollm3 architecture used by llama.cpp GGUF files. We bridge the
    gap by registering smollm3 config/tensor mappings and converting the
    loaded config's model_type to smollm3 after loading.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )

    if "smollm3" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("smollm3")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["smollm3"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    if "llama" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["smollm3"] = TENSOR_PROCESSORS["llama"]

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "llama" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["smollm3"] = GGUF_TO_FAST_CONVERTERS["llama"]

    orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        cfg = result.get("config", {})
        if cfg.get("model_type") == "smollm3":
            cfg.setdefault("no_rope_layer_interval", 4)
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_transformers_smollm3_gguf()
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
    """Available mradermacher/Kai-3B-Distill-Claude-Opus-4.5-GGUF model variants for causal language modeling."""

    KAI_3B_DISTILL_CLAUDE_OPUS_4_5_Q4_K_M_GGUF = (
        "3B_Distill_Claude_Opus_4_5_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """mradermacher/Kai-3B-Distill-Claude-Opus-4.5-GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.KAI_3B_DISTILL_CLAUDE_OPUS_4_5_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Kai-3B-Distill-Claude-Opus-4.5-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KAI_3B_DISTILL_CLAUDE_OPUS_4_5_Q4_K_M_GGUF

    GGUF_FILE = "Kai-3B-Distill-Claude-Opus-4.5.Q4_K_M.gguf"

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
            model="mradermacher Kai-3B-Distill-Claude-Opus-4.5 GGUF",
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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
