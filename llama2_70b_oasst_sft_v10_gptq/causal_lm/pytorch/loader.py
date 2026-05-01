# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama2 70B OASST SFT v10 GPTQ causal language modeling loader
"""
import torch
from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


def _dequantize_gptq_weights(model, pretrained_model_name, bits=4, dtype=torch.bfloat16):
    """Load GPTQ int4-packed weights from safetensors and inject dequantized weights into model.

    transformers 5.x GPTQ quantizer requires optimum+gptqmodel which conflict with the
    tt-xla environment (torch 2.9.1+cpu). Dequantize manually using pure PyTorch instead.
    """
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    import json

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

    shifts = torch.arange(0, 32, bits, dtype=torch.int32)

    def _unpack_int32(packed, bits, out_dim_packed_first=True):
        if out_dim_packed_first:
            K_packed, N = packed.shape
            unpacked = (packed.unsqueeze(1) >> shifts.view(-1, 1)) & ((1 << bits) - 1)
            return unpacked.reshape(K_packed * (32 // bits), N)
        else:
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

                w_int = _unpack_int32(qweight, bits, out_dim_packed_first=True)
                z_int = _unpack_int32(qzeros, bits, out_dim_packed_first=False)

                s = scales[g_idx.long()]
                z = z_int[g_idx.long()].float()
                weight = (w_int.float() - z) * s.float()

                module.weight.data = weight.T.to(dtype)

    return model


class ModelLoader(ForgeModel):
    """Llama2 70B OASST SFT v10 GPTQ model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "TheBloke/Llama2-70B-OASST-SFT-v10-GPTQ"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Llama2-70B-OASST-SFT-v10-GPTQ",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # Strip quantization_config so from_pretrained doesn't invoke the GPTQ
        # quantizer (which requires optimum+gptqmodel, incompatible with this env).
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "cpu",
            "config": config,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()

        # Inject dequantized BF16 weights for all GPTQ-quantized linear layers
        model = _dequantize_gptq_weights(
            model,
            self.model_name,
            bits=4,
            dtype=dtype_override if dtype_override is not None else torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
