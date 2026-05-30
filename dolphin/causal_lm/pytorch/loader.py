# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin 3.0 (Llama 3.2 3B) model loader for causal language modeling.

The upstream checkpoint (``mlx-community/dolphin3.0-llama3.2-3B-4Bit``) ships
weights in Apple MLX's native 4-bit affine-quantized format. Each quantized
linear/embedding weight is stored as three tensors:

    <name>.weight   uint32, packed 4-bit values (8 per word along in-features)
    <name>.scales   fp16,   one per group of ``group_size`` elements
    <name>.biases   fp16,   one per group of ``group_size`` elements

``transformers`` cannot consume this layout (the ``quantization_config`` block
carries no ``quant_method``), so this loader dequantizes the weights back to a
dense floating-point ``LlamaForCausalLM`` state dict — yielding a standard
Llama 3.2 3B model suitable for hardware bring-up.
"""

import json
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

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
from ....tools.utils import cast_input_to_type, pad_inputs


def _dequantize_mlx(packed, scales, biases, group_size, bits):
    """Dequantize one MLX affine-quantized weight tensor to float32.

    Args:
        packed: uint32 tensor [out_features, in_features * bits // 32].
        scales: fp16 tensor [out_features, in_features // group_size].
        biases: fp16 tensor [out_features, in_features // group_size].
        group_size: number of elements sharing a scale/bias.
        bits: bits per quantized value (4 here).

    Returns:
        torch.Tensor: dense [out_features, in_features] float32 weight.
    """
    pack_factor = 32 // bits
    out_features = packed.shape[0]

    # Unpack: value j of each word lives in bits [bits*j, bits*j + bits),
    # lowest nibble first. Convert to int64 first so the shifts are logical
    # (unsigned) and never see a sign bit.
    words = packed.to(torch.int64)
    shifts = (torch.arange(pack_factor, dtype=torch.int64) * bits).view(1, 1, -1)
    mask = (1 << bits) - 1
    q = (words.unsqueeze(-1) >> shifts) & mask          # [O, I//pf, pf]
    q = q.reshape(out_features, -1).to(torch.float32)   # [O, I]

    in_features = q.shape[1]
    num_groups = in_features // group_size
    q = q.reshape(out_features, num_groups, group_size)
    s = scales.to(torch.float32).reshape(out_features, num_groups, 1)
    b = biases.to(torch.float32).reshape(out_features, num_groups, 1)
    return (q * s + b).reshape(out_features, in_features)


class ModelVariant(StrEnum):
    """Available Dolphin model variants for causal LM."""

    DOLPHIN_3_0_LLAMA_3_2_3B_4BIT = "3.0_llama3.2_3B_4bit"


class ModelLoader(ForgeModel):
    """Dolphin 3.0 (Llama 3.2 3B) loader that dequantizes MLX 4-bit weights."""

    _VARIANTS = {
        ModelVariant.DOLPHIN_3_0_LLAMA_3_2_3B_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/dolphin3.0-llama3.2-3B-4Bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOLPHIN_3_0_LLAMA_3_2_3B_4BIT

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with the specified variant.

        Args:
            variant: Optional ModelVariant. If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant. If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        return ModelInfo(
            model="Dolphin",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load the tokenizer for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _build_dequantized_state_dict(self, config):
        """Download the MLX checkpoint and dequantize it into a dense state dict.

        Args:
            config: LlamaConfig carrying the MLX ``quantization_config`` block.

        Returns:
            dict: float32 state dict consumable by ``LlamaForCausalLM``.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        st_path = hf_hub_download(pretrained_model_name, "model.safetensors")

        quant_cfg = getattr(config, "quantization_config", None) or {}
        group_size = quant_cfg.get("group_size", 64)
        bits = quant_cfg.get("bits", 4)

        state_dict = {}
        with safe_open(st_path, framework="pt") as f:
            keys = list(f.keys())
            quant_bases = sorted(k[: -len(".scales")] for k in keys if k.endswith(".scales"))
            handled = set()
            for base in quant_bases:
                w = f.get_tensor(base + ".weight")
                s = f.get_tensor(base + ".scales")
                b = f.get_tensor(base + ".biases")
                state_dict[base + ".weight"] = _dequantize_mlx(
                    w, s, b, group_size, bits
                )
                handled.update(
                    {base + ".weight", base + ".scales", base + ".biases"}
                )
            # Copy through all non-quantized tensors (layernorms, final norm, ...).
            for k in keys:
                if k in handled or k.endswith((".scales", ".biases")):
                    continue
                state_dict[k] = f.get_tensor(k).to(torch.float32)
        return state_dict

    def load_model(self, dtype_override=None):
        """Load and return the dequantized Dolphin (Llama 3.2 3B) model.

        Args:
            dtype_override: Optional torch dtype to cast the model to. If None,
                            the model stays in float32.

        Returns:
            torch.nn.Module: The Llama causal LM instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        # Build a clean (non-quantized) Llama config from the checkpoint's config.
        cfg_path = hf_hub_download(pretrained_model_name, "config.json")
        cfg_dict = json.load(open(cfg_path))
        config = LlamaConfig(**cfg_dict)
        self.config = config

        state_dict = self._build_dequantized_state_dict(config)

        model = LlamaForCausalLM(config)
        model.load_state_dict(state_dict, strict=False)
        # tie_word_embeddings=True: lm_head shares embed_tokens; re-tie to be safe.
        model.tie_weights()
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Dolphin model.

        Args:
            dtype_override: Optional torch dtype applied to the input tensors.
            batch_size: Batch size to replicate the sample to (default 1).

        Returns:
            dict: Input tensors (input_ids, attention_mask) for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_config(self):
        """Load and return the (non-quantized) Llama config for this variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        cfg_path = hf_hub_download(pretrained_model_name, "config.json")
        self.config = LlamaConfig(**json.load(open(cfg_path)))
        return self.config
