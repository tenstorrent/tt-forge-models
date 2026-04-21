# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ReLiK Reader model loader implementation for relation extraction.

ReLiK (Retrieve, Read and Link) is an information-extraction framework whose
reader component combines a transformer encoder (DeBERTa v3 Small here) with
custom heads for entity span detection and relation classification. The
architecture is defined in the model repository itself, so loading requires
``trust_remote_code=True``.
"""

from typing import Optional

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


class ModelVariant(StrEnum):
    """Available ReLiK Reader variants."""

    DEBERTA_V3_SMALL_RE_WIKIPEDIA = "relik-reader-deberta-v3-small-re-wikipedia"


class ModelLoader(ForgeModel):
    """ReLiK Reader model loader for relation extraction."""

    _VARIANTS = {
        ModelVariant.DEBERTA_V3_SMALL_RE_WIKIPEDIA: LLMModelConfig(
            pretrained_model_name="relik-ie/relik-reader-deberta-v3-small-re-wikipedia",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEBERTA_V3_SMALL_RE_WIKIPEDIA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "Michael Jordan was one of the best players in the NBA."
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ReLiK-Reader",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        import torch.nn as nn
        import transformers.modeling_utils

        # PoolerEndLogits was removed from transformers.modeling_utils in transformers>=5.0.
        # Restore it so the model's remote code (modeling_relik.py) can import it.
        if not hasattr(transformers.modeling_utils, "PoolerEndLogits"):

            class PoolerEndLogits(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
                    self.activation = nn.Tanh()
                    self.LayerNorm = nn.LayerNorm(
                        config.hidden_size, eps=config.layer_norm_eps
                    )
                    self.dense_1 = nn.Linear(config.hidden_size, 1)

                def forward(
                    self,
                    hidden_states,
                    start_states=None,
                    start_positions=None,
                    p_mask=None,
                ):
                    assert start_states is not None or start_positions is not None
                    if start_positions is not None:
                        slen, hsz = hidden_states.shape[-2:]
                        start_positions = start_positions[:, None, None].expand(
                            -1, -1, hsz
                        )
                        start_states = hidden_states.gather(-2, start_positions)
                        start_states = start_states.expand(-1, slen, -1)
                    x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
                    x = self.activation(x)
                    x = self.LayerNorm(x)
                    x = self.dense_1(x).squeeze(-1)
                    return x

            transformers.modeling_utils.PoolerEndLogits = PoolerEndLogits

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        return inputs
