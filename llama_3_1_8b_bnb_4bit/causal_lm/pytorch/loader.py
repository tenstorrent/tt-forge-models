# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.1 8B BNB 4-bit model loader implementation for causal language modeling.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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


def _dequantize_bnb_model(model, dtype=torch.bfloat16):
    """Replace all BNB Linear4bit layers with regular nn.Linear using dequantized weights.

    TT hardware has no BNB/CUDA kernel support. Dequantize to bf16 so the model
    can be moved to the XLA device via .to(device).
    """
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F

    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        dq_weight = bnb_F.dequantize_4bit(
            module.weight.data,
            module.weight.quant_state,
            quant_type=module.weight.quant_type,
        ).to(dtype)
        new_linear = nn.Linear(
            dq_weight.shape[1],
            dq_weight.shape[0],
            bias=module.bias is not None,
            dtype=dtype,
        )
        new_linear.weight = nn.Parameter(dq_weight)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(module.bias.to(dtype))
        setattr(parent, attr, new_linear)
    return model


class ModelVariant(StrEnum):
    """Available Llama 3.1 8B BNB 4-bit model variants."""

    LLAMA_3_1_8B_BNB_4BIT = "llama_3_1_8b_bnb_4bit"


class ModelLoader(ForgeModel):
    """Llama 3.1 8B BNB 4-bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_1_8B_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/Llama-3.1-8B-bnb-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_8B_BNB_4BIT

    sample_text = "Hey how are you doing today?"

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
            model="Llama 3.1 8B BNB 4-bit",
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

        # BNB 4-bit quantized model needs CPU device map
        model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model = _dequantize_bnb_model(model, dtype=dtype_override or torch.bfloat16)
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

        if batch_size > 1:
            for i in range(len(sample_inputs)):
                sample_inputs[i] = sample_inputs[i].repeat_interleave(batch_size, dim=0)

        return sample_inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            decoded_output = self.tokenizer.decode(outputs)
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
