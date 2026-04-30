# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLiNER Multitask model loader implementation
"""

from typing import Optional

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


class ModelVariant(StrEnum):
    GLINER_MULTITASK_V1 = "Multitask-v1.0"


class ModelLoader(ForgeModel):
    """GLiNER Multitask model loader implementation."""

    _VARIANTS = {
        ModelVariant.GLINER_MULTITASK_V1: ModelConfig(
            pretrained_model_name="knowledgator/gliner-multitask-v1.0"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLINER_MULTITASK_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GLiNER-Multitask",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the GLiNER Multitask model."""
        import sys

        # The local llm2vec/ model directory is picked up as a Python namespace
        # package, causing gliner's is_module_available("llm2vec") check to
        # incorrectly succeed. Temporarily remove it from sys.path and clear
        # any cached namespace package entries so gliner sees llm2vec as absent.
        project_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if p != project_root]
        cached_llm2vec = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "llm2vec" or k.startswith("llm2vec.")
        }
        try:
            from gliner import GLiNER
            import torch
            import gliner.modeling.utils as _gliner_utils
            import gliner.modeling.base as _gliner_base

            # `torch.where(cond)` (unary form) returns variable-length index
            # tensors whose shape depends on runtime data, which breaks XLA
            # graph tracing in partition_fx_graph_for_cpu_fallback.  Replace
            # with a static scatter_ implementation that routes invalid tokens
            # to a sink slot at index max_text_length and then discards it.
            def _extract_word_embeddings_static(
                token_embeds,
                words_mask,
                attention_mask,
                batch_size,
                max_text_length,
                embed_dim,
                text_lengths,
            ):
                seq_len = words_mask.shape[1]
                valid = words_mask > 0
                target = valid.to(dtype=torch.long) * (words_mask - 1).clamp(
                    min=0
                ) + (~valid).to(dtype=torch.long) * max_text_length
                temp = torch.zeros(
                    batch_size,
                    max_text_length + 1,
                    embed_dim,
                    dtype=token_embeds.dtype,
                    device=token_embeds.device,
                )
                target_3d = target.unsqueeze(-1).expand(batch_size, seq_len, embed_dim)
                temp.scatter_(1, target_3d, token_embeds)
                words_embedding = temp[:, :max_text_length, :].clone()
                aranged = torch.arange(
                    max_text_length,
                    dtype=attention_mask.dtype,
                    device=token_embeds.device,
                ).expand(batch_size, -1)
                mask = aranged < text_lengths
                return words_embedding, mask

            _gliner_utils.extract_word_embeddings = _extract_word_embeddings_static
            _gliner_base.extract_word_embeddings = _extract_word_embeddings_static
        finally:
            sys.path = original_path
            sys.modules.update(cached_llm2vec)

        model_name = self._variant_config.pretrained_model_name
        model = GLiNER.from_pretrained(model_name, **kwargs)
        self.model = model
        return self.model.eval()

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the GLiNER Multitask model.

        Returns a batch suitable for the GLiNER model forward pass.
        """
        text = (
            "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 "
            "to develop and sell BASIC interpreters for the Altair 8800."
        )
        self.text = [text]
        labels = ["founder", "computer", "software", "position", "date"]
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
        decoded = self.model.decoder.decode(
            self.batch["tokens"],
            self.batch["id_to_classes"],
            co_out,
            flat_ner=True,
            threshold=0.5,
            multi_label=False,
        )
        all_entities = []
        for i, spans in enumerate(decoded):
            start_token_idx_to_text_idx = self.all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = self.all_end_token_idx_to_text_idx[i]
            entities = []
            for span in spans:
                start_text_idx = start_token_idx_to_text_idx[span.start]
                end_text_idx = end_token_idx_to_text_idx[span.end]
                ent_details = {
                    "start": start_text_idx,
                    "end": end_text_idx,
                    "text": self.text[i][start_text_idx:end_text_idx],
                    "label": span.entity_type,
                    "score": span.score,
                }
                entities.append(ent_details)
            all_entities.append(entities)
        return all_entities
