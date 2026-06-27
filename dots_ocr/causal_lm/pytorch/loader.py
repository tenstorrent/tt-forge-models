# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr text-decoder (Qwen2 backbone) loader for causal language modeling.

dots.ocr (``DotsOCRForCausalLM``) is a compact document-OCR VLM that pairs a
``DotsVisionTransformer`` image encoder with a plain ``Qwen2ForCausalLM`` text
decoder. This loader exercises the *text-decoder* path only: with text-only
``input_ids`` (no image placeholder tokens) the forward skips the vision tower
entirely and runs the pure Qwen2 decoder stack. The full multimodal path lives
in ``dots_ocr/image_text_to_text/pytorch/loader.py``.
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

# Pin the snapshot so the trust_remote_code modeling/config files are reproducible.
DOTS_OCR_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"


class ModelVariant(StrEnum):
    """Available dots.ocr variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr text-decoder (Qwen2 backbone) loader for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use (for cheap probes).
                        If None, uses the model's default depth.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the dots.ocr model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DotsOCRForCausalLM instance (text-decoder path).
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True, "revision": DOTS_OCR_REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                revision=DOTS_OCR_REVISION,
                trust_remote_code=True,
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return text-only sample inputs (pure Qwen2 decoder path).

        Args:
            dtype_override: Unused for token inputs; kept for interface parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

        # padding=True (not max_length): padded positions get garbage logits that
        # otherwise dominate whole-tensor PCC on this Qwen2 backbone.
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            num_devices % 2 == 0
            and self.config.num_attention_heads % (num_devices // 2) == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads "
                f"across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron column->row tensor-parallel plan for the Qwen2 decoder."""
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
        """Load and return the configuration for the dots.ocr model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
        )
        return self.config
