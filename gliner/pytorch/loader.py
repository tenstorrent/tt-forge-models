# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLiNER model loader implementation
"""

from typing import Optional, Tuple

import torch

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from ...base import ForgeModel


def _xla_extract_word_embeddings(
    token_embeds: torch.Tensor,
    words_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    max_text_length: int,
    embed_dim: int,
    text_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """XLA-compatible replacement for extract_word_embeddings.

    The original uses torch.where() which returns variable-length index tensors,
    followed by in-place scatter with data-dependent indices. Both trigger
    aten._local_scalar_dense on XLA lazy tensors which fails at trace time.

    Uses a one-hot selector built from static tensor shapes and aggregates via
    einsum — all shapes are known at compile time.
    """
    seq_len = words_mask.shape[-1]
    word_indices = torch.arange(
        1, seq_len + 1, device=words_mask.device, dtype=words_mask.dtype
    )
    selector = words_mask.unsqueeze(-1) == word_indices  # (batch, seq_len, seq_len)
    words_embedding = torch.einsum(
        "btw,btd->bwd", selector.to(token_embeds.dtype), token_embeds
    )
    mask = selector.any(dim=1).to(attention_mask.dtype)
    return words_embedding, mask


def _xla_extract_prompt_features(
    class_token_index: int,
    token_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    embed_dim: int,
    embed_ent_token: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """XLA-compatible replacement for extract_prompt_features.

    The original uses torch.where() to find class token positions (variable-length
    result) and scatters embeddings with data-dependent indices — same incompatibility
    with XLA lazy tensors as extract_word_embeddings.

    Uses cumsum to assign 0-indexed entity-type slots to class token positions,
    then builds a one-hot selector and aggregates via einsum.
    """
    class_token_mask = input_ids == class_token_index  # (batch, seq_len)
    cumsum = class_token_mask.long().cumsum(dim=1)
    entity_type_idx = cumsum - 1

    if embed_ent_token:
        source_mask = class_token_mask
    else:
        # Token immediately after each class token; cumsum slot index is unchanged
        # between a class token and its successor so entity_type_idx is still correct.
        source_mask = torch.cat(
            [torch.zeros_like(class_token_mask[:, :1]), class_token_mask[:, :-1]], dim=1
        )

    seq_len = input_ids.shape[-1]
    k_range = torch.arange(
        seq_len, device=input_ids.device, dtype=entity_type_idx.dtype
    )
    selector = source_mask.unsqueeze(-1) & (entity_type_idx.unsqueeze(-1) == k_range)
    prompts_embedding = torch.einsum(
        "btk,btd->bkd", selector.to(token_embeds.dtype), token_embeds
    )
    prompts_embedding_mask = selector.any(dim=1).to(attention_mask.dtype)
    return prompts_embedding, prompts_embedding_mask


class ModelVariant(StrEnum):
    GLINER_LARGEV2 = "Large_v2"
    GLINER_MULTI_V21 = "Multi_v2.1"


class ModelLoader(ForgeModel):
    """GLiNER model loader implementation."""

    _VARIANTS = {
        ModelVariant.GLINER_LARGEV2: ModelConfig(
            pretrained_model_name="urchade/gliner_largev2"
        ),
        ModelVariant.GLINER_MULTI_V21: ModelConfig(
            pretrained_model_name="urchade/gliner_multi-v2.1"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLINER_MULTI_V21

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):

        if variant in [ModelVariant.GLINER_MULTI_V21]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GLiNER",
            variant=variant,
            group=group,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the GLiNER model callable (batch_predict_entities)."""
        from gliner import GLiNER
        from gliner.modeling.layers import LstmSeq2SeqEncoder
        import gliner.modeling.utils as gliner_utils
        import gliner.modeling.base as gliner_base

        # Patch 1: LstmSeq2SeqEncoder
        # pack_padded_sequence + _VF.lstm calls aten._local_scalar_dense to read
        # packed sequence lengths at trace time, failing on XLA lazy tensors.
        #
        # Replaced with an unrolled mask-gated BiLSTM. The backward direction carries
        # h=0 through tail-padding positions so the first real token from the right is
        # processed from zero hidden state, matching packed-sequence behavior. A naive
        # padded LSTM starting from seq_len-1 accumulates (seq_len-N) bias-driven
        # updates before reaching the last real token at N-1, polluting all backward
        # hidden states.
        def _xla_bilstm_forward(self, x, mask, hidden=None):
            lstm = self.lstm
            w_ih_f = lstm.weight_ih_l0
            w_hh_f = lstm.weight_hh_l0
            b_ih_f = lstm.bias_ih_l0
            b_hh_f = lstm.bias_hh_l0
            w_ih_b = lstm.weight_ih_l0_reverse
            w_hh_b = lstm.weight_hh_l0_reverse
            b_ih_b = lstm.bias_ih_l0_reverse
            b_hh_b = lstm.bias_hh_l0_reverse

            batch_size, seq_len, _ = x.shape
            hidden_dim = lstm.hidden_size

            def _cell(xt, h, c, w_ih, w_hh, b_ih, b_hh):
                gates = xt @ w_ih.t() + b_ih + h @ w_hh.t() + b_hh
                i, f, g, o = gates.chunk(4, dim=-1)
                c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h_new = torch.sigmoid(o) * torch.tanh(c_new)
                return h_new, c_new

            h = x.new_zeros(batch_size, hidden_dim)
            c = x.new_zeros(batch_size, hidden_dim)
            fwd_out = []
            for t in range(seq_len):
                h_new, c_new = _cell(x[:, t], h, c, w_ih_f, w_hh_f, b_ih_f, b_hh_f)
                m = mask[:, t].unsqueeze(1).to(x.dtype)
                h = h_new * m + h * (1 - m)
                c = c_new * m + c * (1 - m)
                fwd_out.append(h)

            # Backward: h stays 0 while mask=0 (tail padding), so the first real
            # position from the right sees h=0 — matching packed-sequence behavior.
            h = x.new_zeros(batch_size, hidden_dim)
            c = x.new_zeros(batch_size, hidden_dim)
            bwd_out = [None] * seq_len
            for t in range(seq_len - 1, -1, -1):
                h_new, c_new = _cell(x[:, t], h, c, w_ih_b, w_hh_b, b_ih_b, b_hh_b)
                m = mask[:, t].unsqueeze(1).to(x.dtype)
                h = h_new * m + h * (1 - m)
                c = c_new * m + c * (1 - m)
                bwd_out[t] = h

            fwd = torch.stack(fwd_out, dim=1)
            bwd = torch.stack(bwd_out, dim=1)
            output = torch.cat([fwd, bwd], dim=-1)
            # Zero padding positions to match pad_packed_sequence output.
            return output * mask.unsqueeze(-1).to(x.dtype)

        LstmSeq2SeqEncoder.forward = _xla_bilstm_forward

        # Patch 2 & 3: extract_word_embeddings and extract_prompt_features
        # Both use torch.where() (variable-length output) + in-place scatter with
        # data-dependent indices, which trigger aten._local_scalar_dense on XLA
        # lazy tensors. Replace with static one-hot selector + einsum.
        # Patch in both gliner.modeling.utils (for calls via
        # extract_prompt_features_and_word_embeddings) and gliner.modeling.base
        # (for direct calls from the model class, bound at import time).
        gliner_utils.extract_word_embeddings = _xla_extract_word_embeddings
        gliner_utils.extract_prompt_features = _xla_extract_prompt_features
        gliner_base.extract_word_embeddings = _xla_extract_word_embeddings
        gliner_base.extract_prompt_features = _xla_extract_prompt_features

        model_name = self._variant_config.pretrained_model_name
        model = GLiNER.from_pretrained(model_name, **kwargs)
        self.model = model
        return self.model.eval()

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the GLiNER model with default settings.

        Returns a tuple (texts, labels) suitable for GLiNER.batch_predict_entities.
        """
        text = (
            "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃ'tjɐnu ʁɔ'naldu]; born 5 February 1985) "
            "is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr "
            "and the Portugal national team."
        )
        self.text = text
        labels = ["person", "award", "date", "competitions", "teams"]
        entity_types = list(dict.fromkeys(labels))

        (
            tokens,
            all_start_token_idx_to_text_idx,
            all_end_token_idx_to_text_idx,
        ) = self.model.prepare_inputs(
            texts=[text],
        )
        self.all_start_token_idx_to_text_idx = all_start_token_idx_to_text_idx
        self.all_end_token_idx_to_text_idx = all_end_token_idx_to_text_idx

        input_x = self.model.prepare_base_input(tokens)

        collator = self.model.data_collator_class(
            self.model.config,
            data_processor=self.model.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )

        batch = collator(input_x, entity_types=entity_types)
        self.batch = batch
        return batch

    def post_processing(self, co_out):
        outputs = []
        decoded12 = self.model.decoder.decode(
            self.batch["tokens"],
            self.batch["id_to_classes"],
            co_out,
            flat_ner=True,
            threshold=0.5,
            multi_label=False,
        )
        outputs.extend(decoded12)
        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = self.all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = self.all_end_token_idx_to_text_idx[i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                ent_details = {
                    "start": start_token_idx_to_text_idx[start_token_idx],
                    "end": end_token_idx_to_text_idx[end_token_idx],
                    "text": self.text[i][start_text_idx:end_text_idx],
                    "label": ent_type,
                    "score": ent_score,
                }
                entities.append(ent_details)

            all_entities.append(entities)
        return all_entities
