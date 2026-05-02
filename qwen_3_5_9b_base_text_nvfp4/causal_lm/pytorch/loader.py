# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 9B Base Text NVFP4 model loader implementation for causal language modeling.
"""
import glob
import os

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


def _dequantize_nvfp4_weights(model, pretrained_model_name, target_dtype):
    """Replace randomly-initialized NVFP4-packed weights with properly dequantized BF16.

    The checkpoint stores weights as packed uint8 (2 x e2m1 FP4 per byte) with per-block
    float8_e4m3fn scales and a global float32 outer scale.  transformers ignores these on
    load because it doesn't support the 'modelopt' quant_method.  We use modelopt's own
    NVFP4QTensor to unpack and rescale every affected Linear weight in-place.
    """
    try:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
        from safetensors import safe_open
    except ImportError:
        return  # nvidia-modelopt or safetensors unavailable – leave random weights

    cache_base = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        "models--" + pretrained_model_name.replace("/", "--"),
        "snapshots",
    )
    st_paths = sorted(glob.glob(os.path.join(cache_base, "*", "*.safetensors")))
    if not st_paths:
        return

    # Build param-name → parameter map for fast lookup
    param_map = {name: param for name, param in model.named_parameters()}

    for st_path in st_paths:
        with safe_open(st_path, framework="pt", device="cpu") as st:
            all_keys = set(st.keys())

            for key in all_keys:
                if not key.endswith(".weight"):
                    continue
                scale_key = key[: -len(".weight")] + ".weight_scale"
                scale_2_key = key[: -len(".weight")] + ".weight_scale_2"
                if scale_key not in all_keys or scale_2_key not in all_keys:
                    continue

                packed = st.get_tensor(key)
                if packed.dtype != torch.uint8:
                    continue  # not a packed NVFP4 weight

                scale = st.get_tensor(scale_key)    # float8_e4m3fn [N, K/16]
                scale_2 = st.get_tensor(scale_2_key)  # float32 scalar

                # Reconstruct original shape: each byte encodes 2 values
                original_shape = list(packed.shape)
                original_shape[-1] *= 2

                qtensor = NVFP4QTensor(tuple(original_shape), target_dtype, packed)
                deq = qtensor.dequantize(
                    dtype=target_dtype,
                    scale=scale,
                    double_scale=scale_2,
                    block_sizes={-1: 16},
                )

                # Map checkpoint key to model parameter name.
                # Checkpoint uses 'model.language_model.layers.*'; model uses 'model.layers.*'.
                model_param_name = key
                for old_prefix, new_prefix in [
                    ("model.language_model.", "model."),
                    ("language_model.", ""),
                ]:
                    if model_param_name.startswith(old_prefix):
                        model_param_name = new_prefix + model_param_name[len(old_prefix) :]
                        break

                if model_param_name not in param_map:
                    continue

                with torch.no_grad():
                    param_map[model_param_name].copy_(deq)


class ModelVariant(StrEnum):
    """Available Qwen 3.5 9B Base Text NVFP4 model variants for causal language modeling."""

    QWEN_3_5_9B_BASE_TEXT_NVFP4 = "9B_Base_Text_NVFP4"


class ModelLoader(ForgeModel):
    """Qwen 3.5 9B Base Text NVFP4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_9B_BASE_TEXT_NVFP4: LLMModelConfig(
            pretrained_model_name="osoleve/Qwen3.5-9B-Base-Text-NVFP4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_9B_BASE_TEXT_NVFP4

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
            model="Qwen 3.5 9B Base Text NVFP4",
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

        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
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

        # Weights for NVFP4-quantized layers are randomly re-initialized by from_pretrained
        # because the packed uint8 shapes don't match the model's BF16 shapes.
        # Replace them with properly dequantized BF16 weights.
        _dequantize_nvfp4_weights(
            model, pretrained_model_name, model_kwargs.get("torch_dtype", torch.bfloat16)
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

        # Qwen3_5DynamicCache is not a transformers.Cache subclass; the comparison
        # evaluator calls torch.equal() on it which raises TypeError. Disable caching
        # so outputs contain only logits (no cache object to compare).
        inputs["use_cache"] = False

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
