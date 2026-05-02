# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen 3.5 9B Abliterated MLX 4-bit model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from safetensors import safe_open
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
    """Available Huihui Qwen 3.5 9B Abliterated MLX 4-bit model variants for causal language modeling."""

    HUIHUI_QWEN_3_5_9B_ABLITERATED_MLX_4BIT = "9B_Abliterated_MLX_4bit"


def _mlx_affine_dequantize(weight_u32, scales, biases, group_size=64):
    """Dequantize MLX affine int4 weights packed as uint32 to bfloat16.

    MLX stores 8 uint4 values (nibbles) per uint32 in little-endian nibble order.
    Affine dequantization: float_val = uint4_val * scale + bias, per group.
    Uses numpy to avoid TT XLA __torch_function__ interception on CPU tensors.
    """
    import numpy as np

    out_dim, in_packed = weight_u32.shape
    in_dim = in_packed * 8

    # Unpack uint32 → 8 × uint4 via numpy (bypasses TT XLA CPU tensor override)
    w_np = weight_u32.numpy().view(np.int32)  # reinterpret bits as int32
    shifts = np.arange(8, dtype=np.int32) * 4  # [0, 4, 8, ..., 28]
    int4_vals = ((w_np[:, :, None] >> shifts) & 0xF).reshape(out_dim, in_dim).astype(np.float32)

    # .numpy() on BF16 tensors fails ("Got unsupported ScalarType BFloat16");
    # cast to float32 first — .to() passes through TorchFunctionMode unchanged.
    scales_f32 = scales.to(torch.float32).numpy()  # [out, num_groups]
    biases_f32 = biases.to(torch.float32).numpy()
    scales_exp = np.repeat(scales_f32, group_size, axis=1)  # [out, in_dim]
    biases_exp = np.repeat(biases_f32, group_size, axis=1)

    result_f32 = int4_vals * scales_exp + biases_exp
    return torch.from_numpy(result_f32).to(torch.bfloat16)


def _load_mlx4bit_state_dict(local_path, group_size=64):
    """Load an MLX affine int4 VLM checkpoint, dequantize, and remap to CausalLM keys.

    The checkpoint stores all weights under the 'language_model.' prefix (VLM layout).
    Quantized weights have three entries: .weight (uint32-packed int4), .scales (bf16),
    .biases (bf16). Non-quantized tensors (layer norms, dt_bias, A_log, etc.) are bf16/f32.
    The MLX conv1d stores kernel as [out, K, in_per_group] vs PyTorch's [out, in_per_group, K].
    """
    import os
    st_path = os.path.join(local_path, "model.safetensors")

    with safe_open(st_path, framework="pt") as f:
        all_keys = set(f.keys())

    VLM_PREFIX = "language_model."
    state_dict = {}

    with safe_open(st_path, framework="pt") as f:
        for key in all_keys:
            if key.endswith(".scales") or key.endswith(".biases"):
                continue

            model_key = key[len(VLM_PREFIX):] if key.startswith(VLM_PREFIX) else key

            if key.endswith(".weight"):
                base = key[: -len(".weight")]
                scales_key = f"{base}.scales"
                biases_key = f"{base}.biases"

                tensor = f.get_tensor(key)
                if (
                    tensor.dtype == torch.uint32
                    and scales_key in all_keys
                    and biases_key in all_keys
                ):
                    scales = f.get_tensor(scales_key)
                    biases = f.get_tensor(biases_key)
                    tensor = _mlx_affine_dequantize(tensor, scales, biases, group_size)

                # Conv1d: MLX [out, K, in_per_group] → PyTorch [out, in_per_group, K]
                if "conv1d.weight" in key and tensor.ndim == 3:
                    tensor = tensor.permute(0, 2, 1).contiguous()
            else:
                tensor = f.get_tensor(key)

            state_dict[model_key] = tensor

    return state_dict


class ModelLoader(ForgeModel):
    """Huihui Qwen 3.5 9B Abliterated MLX 4-bit model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN_3_5_9B_ABLITERATED_MLX_4BIT: LLMModelConfig(
            pretrained_model_name="huihui-ai/Huihui-Qwen3.5-9B-abliterated-mlx-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN_3_5_9B_ABLITERATED_MLX_4BIT

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
            model="Huihui Qwen 3.5 9B Abliterated MLX 4-bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import snapshot_download

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        # Load the outer (VLM) config and extract text sub-config
        vlm_config = AutoConfig.from_pretrained(pretrained_model_name)
        text_config = vlm_config.text_config
        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers
        # Disable KV cache: output includes Qwen3_5DynamicCache which tree_map
        # in the comparison evaluator cannot handle (not a Tensor).
        text_config.use_cache = False

        # Create the text-only model with the correct sub-config
        model = Qwen3_5ForCausalLM(text_config).eval()

        # Download checkpoint and dequantize MLX affine int4 weights
        local_path = snapshot_download(pretrained_model_name)
        state_dict = _load_mlx4bit_state_dict(local_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"WARNING: {len(missing)} missing keys in state_dict")

        # Cast to requested dtype
        if dtype_override is not None:
            dtype = getattr(torch, dtype_override) if isinstance(dtype_override, str) else dtype_override
            model = model.to(dtype=dtype)

        self.config = text_config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

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

            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            elif hasattr(layer, "linear_attn"):
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_z.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        vlm_config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        self.config = vlm_config.text_config
        return self.config
