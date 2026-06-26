# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr text-decoder (causal LM) loader implementation.

dots.ocr (`rednote-hilab/dots.ocr`) is a document-OCR VLM whose language model is a
standard Qwen2 decoder (`DotsOCRForCausalLM(Qwen2ForCausalLM)`) with a NaViT vision
tower bolted on. This loader brings up the *text decoder* as a single forward pass:
a text-only prompt (no image tokens, no pixel_values) flows through the Qwen2 decoder.
The vision tower is brought up separately under ``dots_ocr/vision``.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Pin the checkpoint revision so the custom (trust_remote_code) modeling files and
# weights are reproducible across runs.
_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"


class ModelVariant(StrEnum):
    """Available dots.ocr text-decoder variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr text-decoder (Qwen2 causal LM) loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return model metadata for the given variant."""
        return ModelInfo(
            model="dots.ocr text decoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load and cache the tokenizer for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=_REVISION,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the dots.ocr text decoder.

        Loads the full ``DotsOCRForCausalLM`` (trust_remote_code) and drops the vision
        tower so only the Qwen2 decoder weights are resident — the text-only forward
        pass never touches the vision tower.

        Args:
            dtype_override: Optional torch dtype to load weights in (e.g. bfloat16).

        Returns:
            torch.nn.Module: The Qwen2 decoder for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        # Default to float32 on CPU; the runner passes dtype_override=bfloat16 for device.
        model_kwargs = {
            "trust_remote_code": True,
            "revision": _REVISION,
            "dtype": dtype_override if dtype_override is not None else torch.float32,
        }

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs, **kwargs
        )

        # The text-only forward pass never invokes the vision tower; drop it so its
        # ~1.2B params are not loaded onto the device.
        if hasattr(model, "vision_tower"):
            model.vision_tower = torch.nn.Identity()

        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return a tokenized text-only prompt (no image tokens).

        Args:
            dtype_override: Unused for integer token inputs; accepted for interface
                            parity with the runner.
            batch_size: Batch size for the inputs.

        Returns:
            dict: ``{"input_ids", "attention_mask"}`` tensors.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
