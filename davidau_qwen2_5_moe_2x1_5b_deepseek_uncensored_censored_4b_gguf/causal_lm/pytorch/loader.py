# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DavidAU Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B GGUF model loader for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available DavidAU Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B GGUF model variants."""

    QWEN2_5_MOE_2X1_5B_DEEPSEEK_UNCENSORED_CENSORED_4B_Q4_K_M = (
        "Qwen2_5_MOE_2X1_5B_DeepSeek_Uncensored_Censored_4B_Q4_K_M"
    )


class ModelLoader(ForgeModel):
    """DavidAU Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN2_5_MOE_2X1_5B_DEEPSEEK_UNCENSORED_CENSORED_4B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="DavidAU/Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.QWEN2_5_MOE_2X1_5B_DEEPSEEK_UNCENSORED_CENSORED_4B_Q4_K_M
    )

    GGUF_FILE = "Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B-D_AU-Q4_k_m.gguf"

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
            model="DavidAU Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B GGUF",
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

    @staticmethod
    def _fix_moe_config(config, pretrained_model_name: str, gguf_file: str):
        # The GGUF metadata incorrectly inherits Qwen2-MoE-57B-A14B KV values
        # (moe_intermediate_size=1408, shared_expert_intermediate_size=5632).
        # The actual tensors use moe_intermediate_size=8960 for both expert
        # and shared-expert projections. Read the true size from the GGUF.
        if not hasattr(config, "moe_intermediate_size"):
            return config
        from huggingface_hub import hf_hub_download
        import gguf as _gguf

        path = hf_hub_download(pretrained_model_name, filename=gguf_file)
        reader = _gguf.GGUFReader(path)
        moe_size = None
        shared_size = None
        for tensor in reader.tensors:
            name = tensor.name
            # GGUF linear weights are stored column-major: shape = [in, out, ...].
            # blk.0.ffn_gate_exps.weight has PyTorch shape [moe_intermediate, hidden]
            # so GGUF shape is [hidden, moe_intermediate] → shape[1] = moe_intermediate.
            if moe_size is None and (
                ".ffn_gate_exps." in name or ".ffn_up_exps." in name
            ):
                if len(tensor.shape) >= 2:
                    moe_size = int(tensor.shape[1])
            if shared_size is None and (
                ".ffn_gate_shexp." in name or ".ffn_up_shexp." in name
            ):
                if len(tensor.shape) >= 2:
                    shared_size = int(tensor.shape[1])
            if moe_size is not None and shared_size is not None:
                break
        if moe_size is not None:
            config.moe_intermediate_size = moe_size
        if shared_size is not None:
            config.shared_expert_intermediate_size = shared_size
        return config

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        config = self._fix_moe_config(config, pretrained_model_name, self.GGUF_FILE)

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE
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
        pretrained_model_name = self._variant_config.pretrained_model_name
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        self.config = self._fix_moe_config(config, pretrained_model_name, self.GGUF_FILE)
        return self.config
