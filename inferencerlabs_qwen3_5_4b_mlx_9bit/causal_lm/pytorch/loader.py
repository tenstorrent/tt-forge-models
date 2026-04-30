# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5-4B MLX 9-bit model loader for causal language modeling.

inferencerlabs/Qwen3.5-4B-MLX-9bit is an MLX affine-8bit quantized checkpoint.
Its safetensors file stores weights as uint32-packed int8 with per-group bfloat16
scales and biases. The config carries a quantization_config without the quant_method
field that transformers requires, and weight keys are prefixed with "language_model.".

The loader:
1. Creates the Qwen3_5ForCausalLM text model on the meta device (fast).
2. Loads the safetensors checkpoint and dequantizes MLX affine-8bit weights in-place.
3. Remaps keys by stripping the "language_model." prefix.
4. Permutes Conv1d weights from MLX channel-last [out, kernel, in] to PyTorch
   [out, in, kernel] layout.
5. Assigns the resulting state dict into the model and re-ties the embeddings.
"""
import os
import torch
import safetensors.torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
    """Available Qwen3.5-4B MLX 9-bit model variants."""

    QWEN_3_5_4B_MLX_9BIT = "4B_MLX_9bit"


def _dequantize_mlx_affine(raw_sd, group_size=32, target_dtype=torch.bfloat16):
    """Convert MLX affine-8bit safetensors to a standard bfloat16 state dict.

    MLX stores quantized linears as:
      <base>.weight  – uint32, shape [out_f, in_f // 4]  (4 int8s per uint32)
      <base>.scales  – bfloat16, shape [out_f, in_f // group_size]
      <base>.biases  – bfloat16, shape [out_f, in_f // group_size]

    Dequant: w_fp = int8(weight) * scales + biases  (both broadcast over groups).

    Conv1d weights arrive as [out, kernel, in] (MLX channel-last) and are
    permuted to PyTorch's [out, in, kernel].

    All keys have the "language_model." prefix stripped.
    """
    result = {}
    skip = set()

    for key in raw_sd:
        if key in skip:
            continue

        tensor = raw_sd[key]

        new_key = key[len("language_model."):] if key.startswith("language_model.") else key

        if key.endswith(".scales") or key.endswith(".biases"):
            skip.add(key)
            continue

        if tensor.dtype == torch.uint32:
            base = key[: -len(".weight")]
            scales = raw_sd[base + ".scales"]
            biases = raw_sd[base + ".biases"]
            skip.add(base + ".scales")
            skip.add(base + ".biases")

            out_f = tensor.shape[0]
            w_i8 = tensor.view(torch.uint8).reshape(out_f, -1).view(torch.int8).float()

            s = scales.float().repeat_interleave(group_size, dim=1)
            b = biases.float().repeat_interleave(group_size, dim=1)

            result[new_key] = (w_i8 * s + b).to(target_dtype)
        else:
            t = tensor
            if t.ndim == 3:
                t = t.permute(0, 2, 1).contiguous()
            result[new_key] = t

    return result


class ModelLoader(ForgeModel):
    """Qwen3.5-4B MLX 9-bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_4B_MLX_9BIT: LLMModelConfig(
            pretrained_model_name="inferencerlabs/Qwen3.5-4B-MLX-9bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_4B_MLX_9BIT

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
        return ModelInfo(
            model="Qwen3.5-4B MLX 9-bit",
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
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        outer_config = AutoConfig.from_pretrained(pretrained_model_name)
        text_config = outer_config.text_config

        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers
            if hasattr(text_config, "layer_types"):
                text_config.layer_types = text_config.layer_types[: self.num_layers]

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(text_config, dtype=target_dtype)

        safetensors_path = hf_hub_download(pretrained_model_name, "model.safetensors")
        raw_sd = safetensors.torch.load_file(safetensors_path, device="cpu")
        state_dict = _dequantize_mlx_affine(raw_sd, group_size=32, target_dtype=target_dtype)

        model.load_state_dict(state_dict, strict=False, assign=True)
        model.tie_weights()

        # Buffers (inv_freq) are not in the safetensors checkpoint and remain on the
        # meta device after assign-mode load. Re-create the rotary embedding on CPU so
        # those buffers are properly initialized.
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(text_config, device="cpu")

        model.eval()

        self.config = text_config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = self.sample_text
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

    def load_config(self):
        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        self.config = config.text_config
        return self.config
