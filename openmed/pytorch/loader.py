# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed model loader implementation
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
    OPENMED_ZEROSHOT_NER_SPECIES_SMALL = "ZeroShot-NER-Species-Small-166M"
    OPENMED_ZEROSHOT_NER_SPECIES_BASE = "ZeroShot-NER-Species-Base-220M"
    OPENMED_ZEROSHOT_NER_ONCOLOGY_LARGE = "ZeroShot-NER-Oncology-Large-459M"
    OPENMED_ZEROSHOT_NER_BLOODCANCER_LARGE = "ZeroShot-NER-BloodCancer-Large-459M"
    OPENMED_ZEROSHOT_NER_DNA_BASE = "ZeroShot-NER-DNA-Base-220M"


class ModelLoader(ForgeModel):
    """OpenMed model loader implementation."""

    _VARIANTS = {
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Species-Small-166M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_BASE: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Species-Base-220M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_ONCOLOGY_LARGE: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Oncology-Large-459M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_BLOODCANCER_LARGE: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Large-459M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_DNA_BASE: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-DNA-Base-220M"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OpenMed",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the OpenMed GLiNER model."""
        from gliner import GLiNER

        model_name = self._variant_config.pretrained_model_name
        model = GLiNER.from_pretrained(model_name, **kwargs)
        self.model = model
        return self.model.eval()

    _VARIANT_INPUTS = {
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL: {
            "text": "Escherichia coli and Staphylococcus aureus were isolated from the patient samples.",
            "labels": ["SPECIES"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_BASE: {
            "text": "Escherichia coli and Staphylococcus aureus were isolated from the patient samples.",
            "labels": ["SPECIES"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_ONCOLOGY_LARGE: {
            "text": "Mutations in KRAS gene drive oncogenic transformation in colorectal cancer cells.",
            "labels": [
                "Gene_or_gene_product",
                "Cancer",
                "Cell",
                "Simple_chemical",
                "Organ",
                "Tissue",
                "Organism",
            ],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_BLOODCANCER_LARGE: {
            "text": "The patient presented with chronic lymphocytic leukemia symptoms.",
            "labels": ["CL"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_DNA_BASE: {
            "text": "The p53 protein plays a crucial role in tumor suppression.",
            "labels": ["DNA", "RNA", "cell_line", "cell_type", "protein"],
        },
    }

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the OpenMed model.

        Returns a batch suitable for the GLiNER model forward pass.
        """
        variant_input = self._VARIANT_INPUTS[self._variant]
        text = variant_input["text"]
        self.text = [text]
        labels = variant_input["labels"]
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
