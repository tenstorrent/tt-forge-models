# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 18B A3B REAP Coding Heretic i1 GGUF model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _patch_qwen35moe_separate_expert_weights():
    """Patch get_gguf_hf_weights_map to support qwen35moe GGUF files with separate
    gate/up/down expert weight tensors (e.g. i1-quantized models).

    Qwen2MoeTensorProcessor.process() looks up tensor_key_mapping using GGUF names
    without the .weight suffix (e.g. 'blk.0.ffn_gate_exps'), but the standard
    get_gguf_hf_weights_map stores keys with .weight suffix and maps gate_up_proj to
    the fused ffn_gate_up_exps key.  Some i1-quantized GGUF files store expert weights
    separately as ffn_gate_exps / ffn_up_exps / ffn_down_exps rather than fused as
    ffn_gate_up_exps.  This patch adds the without-suffix entries so process() can
    fuse them into the HF gate_up_proj / down_proj tensors.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    _orig = _gguf_utils.get_gguf_hf_weights_map

    def _patched(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        result = _orig(hf_model, processor, model_type, num_layers, qual_name)
        if qual_name != "":
            return result
        mt = (
            model_type
            if model_type is not None
            else (getattr(getattr(hf_model, "config", None), "model_type", None))
        )
        if mt not in ("qwen35moe", "qwen3_5_moe_text", "qwen3_5_moe"):
            return result
        cfg = getattr(hf_model, "config", None)
        if cfg is None:
            return result
        n_layers = (
            num_layers
            if num_layers is not None
            else getattr(cfg, "num_hidden_layers", 0)
        )
        for n in range(n_layers):
            result[
                f"blk.{n}.ffn_gate_exps"
            ] = f"model.layers.{n}.mlp.experts.gate_up_proj.weight"
            result[
                f"blk.{n}.ffn_up_exps"
            ] = f"model.layers.{n}.mlp.experts.gate_up_proj.weight"
            result[
                f"blk.{n}.ffn_down_exps"
            ] = f"model.layers.{n}.mlp.experts.down_proj.weight"
        return result

    _gguf_utils.get_gguf_hf_weights_map = _patched


_patch_qwen35moe_separate_expert_weights()

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
    """Available Qwen 3.5 18B A3B REAP Coding Heretic i1 GGUF model variants for causal language modeling."""

    QWEN_3_5_18B_A3B_REAP_CODING_HERETIC_I1_GGUF = "18B_A3B_REAP_Coding_Heretic_i1_GGUF"


class ModelLoader(ForgeModel):
    """Qwen 3.5 18B A3B REAP Coding Heretic i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_18B_A3B_REAP_CODING_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/qwen3.5-18b-a3b-reap-coding-heretic-v0-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_18B_A3B_REAP_CODING_HERETIC_I1_GGUF

    GGUF_FILE = "qwen3.5-18b-a3b-reap-coding-heretic-v0.i1-Q4_K_M.gguf"

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
            model="Qwen 3.5 18B A3B REAP Coding Heretic i1 GGUF",
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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
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
            enable_thinking=True,
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
            # MoE layers use fused expert weights (3D tensors)
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")
            # Layers have either self_attn (full attention) or linear_attn
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
