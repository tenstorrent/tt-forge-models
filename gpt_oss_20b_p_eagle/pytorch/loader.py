# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS 20B P-EAGLE speculator model loader implementation for speculative decoding.
"""

import torch
from transformers import AutoModelForCausalLM
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GPT-OSS 20B P-EAGLE model variants."""

    GPT_OSS_20B_P_EAGLE = "20B_P_EAGLE"


class ModelLoader(ForgeModel):
    """GPT-OSS 20B P-EAGLE speculator model loader for speculative decoding.

    Loads Amazon's P-EAGLE (Parallel-Drafting EAGLE) draft model for
    openai/gpt-oss-20b. The draft model is a 4-layer LlamaForCausalLM that
    generates multiple draft tokens per forward pass to accelerate inference
    of the target verifier model.
    """

    _VARIANTS = {
        ModelVariant.GPT_OSS_20B_P_EAGLE: ModelConfig(
            pretrained_model_name="amazon/GPT-OSS-20B-P-EAGLE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B_P_EAGLE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GPT-OSS 20B P-EAGLE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GPT-OSS 20B P-EAGLE speculator model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The P-EAGLE speculator model instance.
        """
        cfg = self._variant_config

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the P-EAGLE speculator model.

        The P-EAGLE speculator shares the gpt-oss-20b tokenizer/vocabulary
        (vocab_size=201088) but does not ship its own tokenizer files, so
        sample input_ids are generated directly.

        Args:
            dtype_override: Unused; included for API compatibility.

        Returns:
            dict: Input tensors containing input_ids for the speculator.
        """
        seq_len = 8
        vocab_size = 201088  # shared with openai/gpt-oss-20b

        torch.manual_seed(42)
        input_ids = torch.randint(0, vocab_size, (1, seq_len))

        return {"input_ids": input_ids}
