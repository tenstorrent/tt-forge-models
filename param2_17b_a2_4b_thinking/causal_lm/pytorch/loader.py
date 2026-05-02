# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bharatgenai/Param2-17B-A2.4B-Thinking model loader implementation for causal language modeling.
"""
import importlib
import sys
from typing import Optional

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# transformers 5.x removed is_torch_fx_available; the remote model code still imports it.
if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
    def _is_torch_fx_available():
        return False
    transformers.utils.import_utils.is_torch_fx_available = _is_torch_fx_available
    sys.modules["transformers.utils.import_utils"].__dict__["is_torch_fx_available"] = _is_torch_fx_available


def _static_moe_infer(self, x, topk_ids, topk_weight):
    """Static per-expert masked matmul — no D2H / numpy, traceable by XLA."""
    final_out = torch.zeros(x.shape[0], x.shape[-1], dtype=x.dtype, device=x.device)
    for expert_idx in range(len(self.experts)):
        mask = topk_ids == expert_idx  # [T, k]
        weight = (topk_weight * mask.to(topk_weight.dtype)).sum(dim=-1)  # [T]
        masked_x = x * mask.any(dim=-1).unsqueeze(-1).to(x.dtype)
        expert_out = self.experts[expert_idx](masked_x)
        final_out = final_out + expert_out * weight.unsqueeze(-1).to(expert_out.dtype)
    return final_out


def _patch_model_class(pretrained_model_name: str) -> type:
    # transformers 5.x removed 'default' from ROPE_INIT_FUNCTIONS; this model uses it.
    if "default" not in ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
            base = getattr(config, "rope_theta", 10000.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            partial_rotary_factor = 1.0
            if hasattr(config, "rope_scaling") and config.rope_scaling:
                partial_rotary_factor = config.rope_scaling.get("partial_rotary_factor", 1.0)
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    # transformers 5.x expects _tied_weights_keys as {target: source} dict; model has a list.
    cls = get_class_from_dynamic_module(
        "modeling_param2moe.Param2MoEForCausalLM", pretrained_model_name
    )
    if isinstance(cls._tied_weights_keys, list):
        cls._tied_weights_keys = {"lm_head.weight": "model.word_embeddings.weight"}

    # moe_infer uses .cpu().numpy() D2H which hangs / crashes under XLA tracing; replace with
    # static per-expert masked matmul.
    model_module = importlib.import_module(cls.__module__)
    model_module.Param2MoESparseMoeBlock.moe_infer = _static_moe_infer

    return cls


class ModelVariant(StrEnum):
    """Available Param2-17B-A2.4B-Thinking model variants."""

    PARAM2_17B_A2_4B_THINKING = "param2_17b_a2_4b_thinking"


class ModelLoader(ForgeModel):
    """Param2-17B-A2.4B-Thinking model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PARAM2_17B_A2_4B_THINKING: ModelConfig(
            pretrained_model_name="bharatgenai/Param2-17B-A2.4B-Thinking",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PARAM2_17B_A2_4B_THINKING

    sample_messages = [
        {"role": "system", "content": "You are helpful assistant."},
        {"role": "user", "content": "What is the BharatGen Mission?"},
    ]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Param2-17B-A2.4B-Thinking",
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
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        model_cls = _patch_model_class(pretrained_model_name)

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model_kwargs = {
            "use_cache": False,
            "torch_dtype": model_dtype,
            "trust_remote_code": True,
        }
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = model_cls.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        text = self.tokenizer.apply_chat_template(
            self.sample_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
