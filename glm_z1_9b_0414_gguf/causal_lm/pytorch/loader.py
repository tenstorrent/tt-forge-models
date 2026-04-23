# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-Z1-9B-0414 GGUF model loader for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer, Glm4ForCausalLM

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


def _register_glm4_gguf_support():
    # transformers GGUF loading does not support glm4 natively; register the
    # config field mapping so AutoConfig.from_pretrained(..., gguf_file=...)
    # can extract model hyperparameters from the GGUF metadata.
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
    from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES

    if "glm4" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["glm4"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": "head_dim",
            "vocab_size": "vocab_size",
        }
        GGUF_SUPPORTED_ARCHITECTURES.append("glm4")


_register_glm4_gguf_support()

# Original (non-GGUF) model used for tokenizer loading; GGUF tokenizer
# conversion for glm4 is not yet supported by transformers.
_TOKENIZER_SOURCE = "THUDM/GLM-Z1-9B-0414"


class ModelVariant(StrEnum):
    """Available GLM-Z1-9B-0414 GGUF model variants."""

    GLM_Z1_9B_0414_GGUF = "Z1_9B_0414_GGUF"


class ModelLoader(ForgeModel):
    """GLM-Z1-9B-0414 GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GLM_Z1_9B_0414_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/THUDM_GLM-Z1-9B-0414-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_Z1_9B_0414_GGUF

    GGUF_FILE = "THUDM_GLM-Z1-9B-0414-Q4_K_M.gguf"

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
            model="GLM-Z1-9B-0414 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # Load from the original (non-GGUF) repo because transformers does not
        # support GGUF tokenizer conversion for the glm4 architecture yet.
        self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_SOURCE)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if self.config is None:
            self.load_config()

        config = self.config
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model_kwargs = {}
        model_kwargs |= kwargs

        # GGUF weight loading for glm4 is not supported by transformers; the
        # model is instantiated with random weights (suitable for compile-only).
        # Glm4ForCausalLM.__init__ does not accept torch_dtype, so cast after.
        model = Glm4ForCausalLM(config, **model_kwargs).eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
