# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LXMERT model loader implementation for multimodal feature extraction.
"""
import torch
from transformers import AutoTokenizer, LxmertModel
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
    """Available LXMERT model variants."""

    BASE_UNCASED = "base-uncased"


class ModelLoader(ForgeModel):
    """LXMERT model loader implementation for multimodal feature extraction."""

    _VARIANTS = {
        ModelVariant.BASE_UNCASED: ModelConfig(
            pretrained_model_name="unc-nlp/lxmert-base-uncased",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_UNCASED

    text = "Where is the cat sitting?"
    num_visual_features = 36
    visual_feat_dim = 2048
    visual_pos_dim = 4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LXMERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LxmertModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        text_inputs = self.tokenizer(
            self.text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=20,
        )

        # LXMERT expects ROI-pooled object features from an external Faster-RCNN
        # model, which HuggingFace's transformers library does not provide. Use
        # deterministic random tensors of the expected shapes instead.
        generator = torch.Generator().manual_seed(0)
        visual_feats = torch.rand(
            1,
            self.num_visual_features,
            self.visual_feat_dim,
            generator=generator,
        )
        visual_pos = torch.rand(
            1,
            self.num_visual_features,
            self.visual_pos_dim,
            generator=generator,
        )

        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "token_type_ids": text_inputs["token_type_ids"],
            "visual_feats": visual_feats,
            "visual_pos": visual_pos,
        }

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["visual_feats"] = inputs["visual_feats"].to(dtype_override)
            inputs["visual_pos"] = inputs["visual_pos"].to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
