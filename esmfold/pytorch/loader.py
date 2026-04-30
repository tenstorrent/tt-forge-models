# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ESMFold (facebook/esmfold_v1) model loader implementation for protein structure prediction.
"""

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.modeling_esm import RotaryEmbedding
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
    """Available ESMFold model variants."""

    ESMFOLD_V1 = "facebook/esmfold_v1"


class ModelLoader(ForgeModel):
    """ESMFold model loader implementation for protein structure prediction."""

    _VARIANTS = {
        ModelVariant.ESMFOLD_V1: ModelConfig(
            pretrained_model_name="facebook/esmfold_v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ESMFOLD_V1

    # Short protein sequence for testing
    sample_sequence = "MGSSHHHHHHSSGLVPRGSHM"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ESMFold",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    @staticmethod
    def _patch_rotary_embedding():
        """Remove stateful cosine/sine caching from RotaryEmbedding.

        _update_cos_sin_tables mutates _seq_len_cached/_cos_cached/_sin_cached on the
        module instance.  torch.compile/dynamo restarts the trace when it encounters
        the inv_freq buffer access (via the __getattr__ fallback path), and on that
        second pass the cached state makes the control-flow branch differently,
        producing a SpeculationLogDivergence.  The cache is a pure performance
        optimisation; removing it yields identical numerical results.
        """

        def _stateless_update(self, x, seq_dimension=2):
            # Access inv_freq via _buffers dict (not via self.inv_freq attribute)
            # because inspect.getattr_static cannot find nn.Module buffers stored
            # in _buffers, so dynamo's _getattr_static raises AttributeError which
            # triggers a trace restart and SpeculationLogDivergence.
            inv_freq = self._buffers["inv_freq"]
            seq_len = x.shape[seq_dimension]
            t = torch.arange(seq_len, device=x.device).to(dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

        RotaryEmbedding._update_cos_sin_tables = _stateless_update

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self._patch_rotary_embedding()

        model = EsmForProteinFolding.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_sequence,
            return_tensors="pt",
            add_special_tokens=False,
        )

        return inputs
