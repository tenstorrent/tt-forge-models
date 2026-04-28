# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM model loader implementation
"""


from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional


class ModelLoader(ForgeModel):
    """MiniCPM model loader implementation."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "openbmb/MiniCPM-2B-sft-bf16"
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="MiniCPM",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MiniCPM model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MiniCPM model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load pre-trained model from HuggingFace
        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Transformers 5.x normalizes rope_scaling=null to {'rope_type': 'default', ...},
        # but the custom model code (trust_remote_code) expects None or {'type': ...}.
        # Always load config and translate the rope_scaling format.
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        if isinstance(config.rope_scaling, dict) and "type" not in config.rope_scaling:
            rope_type = config.rope_scaling.get("rope_type", "default")
            if rope_type == "default":
                config.rope_scaling = None
            else:
                config.rope_scaling = dict(config.rope_scaling, type=rope_type)
        config.use_cache = False
        # transformers>=5.0 expects _tied_weights_keys as dict; this model uses list format
        # (transformers 4.x). Disabling tie_word_embeddings skips the incompatible code path.
        config.tie_word_embeddings = False
        model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        # lm_head.weight is not stored in the checkpoint because it is tied to
        # embed_tokens.weight. After disabling tie_word_embeddings, it is randomly
        # initialized. Re-tie it here to restore correct model behaviour.
        model.lm_head.weight = model.model.embed_tokens.weight

        # transformers>=5.0 uses meta device during from_pretrained, leaving
        # non-persistent buffers (inv_freq, cos_cached, sin_cached) uninitialized.
        # Reinitialize them explicitly to get valid RoPE values.
        import torch
        for module in model.modules():
            if (
                hasattr(module, "inv_freq")
                and hasattr(module, "base")
                and hasattr(module, "dim")
                and hasattr(module, "_set_cos_sin_cache")
            ):
                inv_freq = 1.0 / (
                    module.base
                    ** (torch.arange(0, module.dim, 2).float() / module.dim)
                )
                module.register_buffer("inv_freq", inv_freq, persistent=False)
                module._set_cos_sin_cache(
                    seq_len=module.max_seq_len_cached,
                    device=inv_freq.device,
                    dtype=torch.float32,
                )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the MiniCPM model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        text = "What is gravity?"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded answer text
        """
        if self.tokenizer is None:
            self.load_model()

        next_token_logits = outputs.logits[:, -1]
        next_tokens = next_token_logits.softmax(dim=-1).argmax(dim=-1)
        return [self.tokenizer.decode([token.item()]) for token in next_tokens]
