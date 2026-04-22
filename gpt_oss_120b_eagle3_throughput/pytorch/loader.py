# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS 120B EAGLE3 throughput speculator model loader implementation for speculative decoding.
"""

import torch
from transformers import AutoModel
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
    """Available GPT-OSS 120B EAGLE3 throughput speculator model variants."""

    GPT_OSS_120B_EAGLE3_THROUGHPUT = "120B_Eagle3_throughput"


class ModelLoader(ForgeModel):
    """GPT-OSS 120B EAGLE3 throughput speculator model loader for speculative decoding.

    Loads the NVIDIA GPT-OSS-120B EAGLE3 throughput speculator draft model, which
    accelerates inference of the openai/gpt-oss-120b verifier model via speculative
    decoding.
    """

    _VARIANTS = {
        ModelVariant.GPT_OSS_120B_EAGLE3_THROUGHPUT: ModelConfig(
            pretrained_model_name="nvidia/gpt-oss-120b-Eagle3-throughput",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_120B_EAGLE3_THROUGHPUT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GPT-OSS 120B EAGLE3 throughput",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GPT-OSS 120B EAGLE3 throughput speculator model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The EAGLE3 speculator model instance.
        """
        cfg = self._variant_config

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            cfg.pretrained_model_name,
            ignore_mismatched_sizes=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the EAGLE3 speculator model.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.

        Returns:
            dict: Input tensors for the speculator.
        """
        seq_len = 1
        torch.manual_seed(42)
        input_ids = torch.randint(0, 201088, (1, seq_len))
        return {"input_ids": input_ids}
