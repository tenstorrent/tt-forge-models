# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.2 3B GPTQ 4-bit 128 group size model loader implementation for causal language modeling.
"""
import torch
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


def _dequantize_gptq_weights(model, pretrained_model_name, bits=4, dtype=torch.bfloat16):
    """Load GPTQ int4-packed weights from safetensors and inject dequantized weights into model.

    transformers 5.x GPTQ quantizer requires optimum+gptqmodel, which conflict with the
    tt-xla environment (torch 2.9.1+cpu). Dequantize manually using pure PyTorch instead.
    """
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    import json
    import os

    # Locate safetensors file(s); handle both single-file and sharded checkpoints
    cache_root = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )
    try:
        index_path = hf_hub_download(
            pretrained_model_name, "model.safetensors.index.json"
        )
        with open(index_path) as f:
            index = json.load(f)
        unique_shards = sorted(set(index["weight_map"].values()))
        shard_paths = [
            hf_hub_download(pretrained_model_name, s) for s in unique_shards
        ]
    except Exception:
        shard_paths = [hf_hub_download(pretrained_model_name, "model.safetensors")]

    # Build a lookup: prefix -> open safe_open handle (opened lazily per shard)
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)

    def _unpack_int32(packed, bits, out_dim_packed_first=True):
        """Unpack int32 tensor along its last dim (or first) into int4 values."""
        if out_dim_packed_first:
            # packed: [K_packed, N] -> [K_packed, 8, N] -> [K, N]
            K_packed, N = packed.shape
            unpacked = (packed.unsqueeze(1) >> shifts.view(-1, 1)) & ((1 << bits) - 1)
            return unpacked.reshape(K_packed * (32 // bits), N)
        else:
            # packed: [G, N_packed] -> [G, N_packed, 8] -> [G, N]
            G, N_packed = packed.shape
            unpacked = (packed.unsqueeze(2) >> shifts.view(1, 1, -1)) & (
                (1 << bits) - 1
            )
            return unpacked.reshape(G, N_packed * (32 // bits))

    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            all_keys = set(sf.keys())
            for mod_name, module in model.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                qweight_key = f"{mod_name}.qweight"
                if qweight_key not in all_keys:
                    continue

                qweight = sf.get_tensor(qweight_key)
                qzeros = sf.get_tensor(f"{mod_name}.qzeros")
                scales = sf.get_tensor(f"{mod_name}.scales")
                g_idx = sf.get_tensor(f"{mod_name}.g_idx")

                # Unpack int32 packed tensors to int4 values
                w_int = _unpack_int32(qweight, bits, out_dim_packed_first=True)
                z_int = _unpack_int32(qzeros, bits, out_dim_packed_first=False)

                # Dequantize: weight = (w - zeros) * scales, indexed by g_idx
                s = scales[g_idx.long()]  # [K, N] float16
                z = z_int[g_idx.long()].float()  # [K, N]
                weight = (w_int.float() - z) * s.float()  # [K, N]

                # nn.Linear.weight is [out_features, in_features] = [N, K]
                module.weight.data = weight.T.to(dtype)

    return model


class ModelVariant(StrEnum):
    """Available Llama 3.2 3B GPTQ 4-bit 128 group size model variants for causal language modeling."""

    LLAMA_3_2_3B_GPTQ_4BIT_128G = "Llama-3.2-3B_4bits_128group_size"


class ModelLoader(ForgeModel):
    """Llama 3.2 3B GPTQ 4-bit 128 group size model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_3B_GPTQ_4BIT_128G: LLMModelConfig(
            pretrained_model_name="sliuau/Llama-3.2-3B_4bits_128group_size",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_3B_GPTQ_4BIT_128G

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
        return ModelInfo(
            model="Llama 3.2 3B GPTQ 4-bit 128 group size",
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

        # Strip quantization_config so from_pretrained doesn't invoke the GPTQ
        # quantizer (which requires optimum+gptqmodel, incompatible with this env).
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model_kwargs = {"device_map": "cpu", "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Inject dequantized BF16 weights for all GPTQ-quantized linear layers
        model = _dequantize_gptq_weights(
            model,
            pretrained_model_name,
            bits=4,
            dtype=dtype_override if dtype_override is not None else torch.bfloat16,
        )

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
            self._variant_config.pretrained_model_name
        )
        return self.config
