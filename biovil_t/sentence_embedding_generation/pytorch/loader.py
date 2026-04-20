# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BioViL-T model loader for radiology text embedding generation.

microsoft/BiomedVLP-BioViL-T is a biomedical vision-language model trained on
chest X-rays and radiology reports. The HuggingFace checkpoint exposes the
CXR-BERT text encoder with a projection head via trust_remote_code; this
loader drives the text encoder's forward pass to produce contextual token
embeddings that the joint image-text latent space is built on.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available BioViL-T model variants."""

    BIOMEDVLP_BIOVIL_T = "microsoft/BiomedVLP-BioViL-T"


class ModelLoader(ForgeModel):
    """BioViL-T model loader for radiology text embedding generation."""

    _VARIANTS = {
        ModelVariant.BIOMEDVLP_BIOVIL_T: LLMModelConfig(
            pretrained_model_name="microsoft/BiomedVLP-BioViL-T",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BIOMEDVLP_BIOVIL_T

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BioViL-T",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if sentence is None:
            sentence = "No pleural effusion or pneumothorax is seen."

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            sentence,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if hasattr(fwd_output, "logits") and fwd_output.logits is not None:
            tensors.append(fwd_output.logits.flatten())
        if (
            hasattr(fwd_output, "cls_projected_embedding")
            and fwd_output.cls_projected_embedding is not None
        ):
            tensors.append(fwd_output.cls_projected_embedding.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
