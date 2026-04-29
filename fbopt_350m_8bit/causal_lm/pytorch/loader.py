# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
fbopt-350m-8bit model loader implementation for causal language modeling.
"""

import torch
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
    """Available fbopt-350m-8bit model variants."""

    FBOPT_350M_8BIT = "350m-8bit"


class ModelLoader(ForgeModel):
    """fbopt-350m-8bit model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.FBOPT_350M_8BIT: LLMModelConfig(
            pretrained_model_name="yec019/fbopt-350m-8bit",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.FBOPT_350M_8BIT

    # Sample text for causal LM
    sample_text = "My name is Thomas and my main"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="fbopt-350m-8bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    @staticmethod
    def _dequantize_bnb_state_dict(path, dtype):
        """Load a bitsandbytes int8 safetensors file and dequantize to the given dtype.

        Bitsandbytes stores: weight (int8, shape [out, in]) and SCB (float32, shape [out])
        where SCB is the per-row absolute max. Dequantization: W = W_int8 * (SCB / 127).
        """
        from safetensors import safe_open

        raw = {}
        scb = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".SCB"):
                    scb[k[: -len(".SCB")] + ".weight"] = f.get_tensor(k)
                else:
                    raw[k] = f.get_tensor(k)

        state_dict = {}
        for k, v in raw.items():
            if v.dtype == torch.int8 and k in scb:
                scale = scb[k]
                state_dict[k] = (v.float() * (scale.view(-1, 1) / 127.0)).to(dtype)
            else:
                state_dict[k] = v.to(dtype)
        return state_dict

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the fbopt-350m-8bit model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(pretrained_model_name)

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        # The checkpoint uses bitsandbytes int8 quantization which requires CUDA.
        # Dequantize the weights manually so the model can run on TT hardware.
        _qc = getattr(config, "quantization_config", None)
        _quant_method = (
            _qc.get("quant_method")
            if isinstance(_qc, dict)
            else getattr(_qc, "quant_method", None)
        )
        if _quant_method == "bitsandbytes":
            config.quantization_config = None

            from huggingface_hub import hf_hub_download

            safetensors_path = hf_hub_download(
                pretrained_model_name, "model.safetensors"
            )
            state_dict = self._dequantize_bnb_state_dict(safetensors_path, dtype)

            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
            model.load_state_dict(state_dict, strict=False)
        else:
            model_kwargs = {"torch_dtype": dtype, "device_map": "cpu"}
            model_kwargs |= kwargs
            model_kwargs["config"] = config
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model [input_ids, attention_mask].
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_tokens = self.tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return [input_tokens["input_ids"], input_tokens["attention_mask"]]
