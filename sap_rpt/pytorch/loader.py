# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAP RPT-1-OSS model loader for tabular classification.

SAP RPT-1-OSS (formerly ConTextTab) is a tabular foundation model for
in-context learning on tabular data. It uses 2D attention (cross-column +
cross-row) with cell embeddings derived from sentence transformers.
"""
import torch
import torch.nn as nn
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
    """Available SAP RPT model variants."""

    SAP_RPT_1_OSS = "SAP/sap-rpt-1-oss"


class SapRptWrapper(nn.Module):
    """XLA-friendly wrapper around RPT model components.

    Pre-computes the context attention mask externally (avoiding
    data-dependent boolean indexing that breaks dynamo tracing) and
    inlines the forward pass using only XLA-compatible ops.
    """

    def __init__(self, rpt_model):
        super().__init__()
        self.embeddings = rpt_model.embeddings
        self.in_context_encoder = rpt_model.in_context_encoder
        self.dense_classif = rpt_model.dense_classif
        self.output_head_classif = rpt_model.output_head_classif

    def forward(
        self,
        column_embeddings,
        text_embeddings,
        date_year_month_day_weekday,
        target,
        target_delta,
        number_normalized,
        attention_mask,
    ):
        from transformers.activations import gelu

        data = {
            "column_embeddings": column_embeddings,
            "text_embeddings": text_embeddings,
            "date_year_month_day_weekday": date_year_month_day_weekday,
            "target": target,
            "target_delta": target_delta,
            "number_normalized": number_normalized,
        }

        input_embeds = self.embeddings(data, False)
        attention_mask = attention_mask.to(dtype=input_embeds.dtype)

        for layer in self.in_context_encoder:
            input_embeds = layer(input_embeds, attention_mask)

        target_column_output = input_embeds[:, -1]

        out = self.dense_classif(target_column_output)
        out = gelu(out)
        out = self.output_head_classif(out)
        return out


class ModelLoader(ForgeModel):
    """SAP RPT-1-OSS model loader for tabular classification."""

    _VARIANTS = {
        ModelVariant.SAP_RPT_1_OSS: ModelConfig(
            pretrained_model_name="SAP/sap-rpt-1-oss",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAP_RPT_1_OSS

    NUM_TRAIN_ROWS = 4
    NUM_TEST_ROWS = 2
    NUM_FEATURE_COLS = 3
    EMBEDDING_DIM = 384

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SAP-RPT-1-OSS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_context_attention_mask(self, target):
        """Build context attention mask using XLA-friendly operations.

        Context rows (target > -99) can be attended by all rows.
        Query rows (target <= -99) can only attend to themselves.
        """
        num_rows = target.shape[0]
        identity = torch.eye(num_rows, dtype=torch.float32)
        context_cols = (target > -99).float().unsqueeze(0)
        mask = torch.where(context_cols > 0, torch.ones_like(identity), identity)
        return (1.0 - mask) * torch.finfo(torch.float32).min

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load SAP RPT-1-OSS model wrapped for tensor-based inference."""
        from sap_rpt_oss.constants import ModelSize
        from sap_rpt_oss.model.torch_model import RPT

        model = RPT(
            ModelSize.base,
            regression_type="l2",
            classification_type="cross-entropy",
            checkpointing_segments=0,
        )
        model.eval()

        wrapper = SapRptWrapper(model)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        """Prepare sample inputs for the SAP RPT model."""
        num_rows = self.NUM_TRAIN_ROWS + self.NUM_TEST_ROWS
        num_cols = self.NUM_FEATURE_COLS + 1

        dtype = dtype_override if dtype_override is not None else torch.float32

        column_embeddings = torch.randn(
            num_cols, self.EMBEDDING_DIM, dtype=torch.float16
        )
        text_embeddings = torch.zeros(
            num_rows, num_cols, self.EMBEDDING_DIM, dtype=torch.float16
        )
        date_year_month_day_weekday = torch.zeros(
            num_rows, num_cols, 4, dtype=torch.int64
        )

        target = torch.zeros(num_rows, dtype=dtype)
        target[: self.NUM_TRAIN_ROWS] = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype)
        target[self.NUM_TRAIN_ROWS :] = -100.0

        target_delta = torch.zeros(num_rows, dtype=dtype)
        number_normalized = torch.full((num_rows, num_cols), -100.0, dtype=dtype)

        attention_mask = self._build_context_attention_mask(target)

        return [
            column_embeddings,
            text_embeddings,
            date_year_month_day_weekday,
            target,
            target_delta,
            number_normalized,
            attention_mask,
        ]
