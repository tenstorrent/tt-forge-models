# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral-Small-4-119B-2603 MLX 9-bit model loader for causal language modeling.
"""
import torch
from transformers import AutoConfig, AutoTokenizer, Mistral4ForCausalLM
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
    """Available Mistral-Small-4-119B-2603 MLX 9-bit model variants."""

    MISTRAL_SMALL_4_119B_2603_MLX_9BIT = "119B_2603_MLX_9bit"


class ModelLoader(ForgeModel):
    """Mistral-Small-4-119B-2603 MLX 9-bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_4_119B_2603_MLX_9BIT: LLMModelConfig(
            pretrained_model_name="inferencerlabs/Mistral-Small-4-119B-2603-MLX-9bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_4_119B_2603_MLX_9BIT

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Mistral-Small-4-119B-2603 MLX 9-bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # The HF repo config is mistral3 (multimodal). Extract the text_config
        # (mistral4/MoE) and build from random weights — the model uses MLX-specific
        # 9-bit affine quantization that PyTorch cannot load from pretrained.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        text_config = config.text_config

        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers

        model = Mistral4ForCausalLM(text_config).eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
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
            # Fused experts tensor: [num_experts, intermediate, hidden]
            shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
            shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            # Shared expert uses standard MLP layout
            shard_specs[mlp.shared_experts.gate_proj.weight] = ("model", "batch")
            shard_specs[mlp.shared_experts.up_proj.weight] = ("model", "batch")
            shard_specs[mlp.shared_experts.down_proj.weight] = ("batch", "model")

            # Mistral4 uses low-rank (LoRA-style) QKV projections; shard B matrices
            shard_specs[layer.self_attn.q_b_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.kv_b_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
        self.config = config.text_config
        return self.config
