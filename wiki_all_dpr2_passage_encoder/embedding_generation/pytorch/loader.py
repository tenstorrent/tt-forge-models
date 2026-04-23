# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wiki-All DPR2 Passage Encoder model loader for embedding generation.
"""
from typing import Optional

import flax.serialization
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Wiki-All DPR2 Passage Encoder model variants."""

    WIKI_ALL_8_4_MULTI_DPR2_PASSAGE_ENCODER = (
        "castorini/wiki-all-8-4-multi-dpr2-passage-encoder"
    )


class ModelLoader(ForgeModel):
    """Wiki-All DPR2 Passage Encoder model loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.WIKI_ALL_8_4_MULTI_DPR2_PASSAGE_ENCODER: LLMModelConfig(
            pretrained_model_name="castorini/wiki-all-8-4-multi-dpr2-passage-encoder",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WIKI_ALL_8_4_MULTI_DPR2_PASSAGE_ENCODER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wiki-All-DPR2-Passage-Encoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        return self.tokenizer

    @staticmethod
    def _load_from_flax(model_name: str) -> dict:
        """Convert flax_model.msgpack to a PyTorch state dict.

        transformers 5.x removed from_flax support; this replicates the conversion.
        """
        flax_file = hf_hub_download(model_name, "flax_model.msgpack")
        with open(flax_file, "rb") as f:
            flax_params = flax.serialization.msgpack_restore(f.read())

        def _flatten(d, prefix=""):
            items = {}
            for k, v in d.items():
                full_key = f"{prefix}/{k}" if prefix else k
                if hasattr(v, "keys"):
                    items.update(_flatten(v, full_key))
                else:
                    items[full_key] = np.array(v)
            return items

        flat = _flatten(flax_params)
        state_dict = {}
        for flax_key, arr in flat.items():
            pt_key = flax_key.replace("/", ".")
            if pt_key.endswith(".kernel"):
                pt_key = pt_key[: -len(".kernel")] + ".weight"
                tensor = torch.from_numpy(arr)
                if tensor.ndim == 2:
                    tensor = tensor.T
            elif pt_key.endswith(".embedding") or pt_key.endswith(".scale"):
                suffix = ".embedding" if pt_key.endswith(".embedding") else ".scale"
                pt_key = pt_key[: -len(suffix)] + ".weight"
                tensor = torch.from_numpy(arr)
            else:
                tensor = torch.from_numpy(arr)
            state_dict[pt_key] = tensor
        return state_dict

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_config(config)

        state_dict = self._load_from_flax(model_name)
        model.load_state_dict(state_dict, strict=True)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        passage = (
            "Dense passage retrieval is a dense retrieval method used for "
            "open-domain question answering over a Wikipedia corpus."
        )

        max_length = getattr(self._variant_config, "max_length", 512)

        inputs = self.tokenizer(
            passage,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            last_hidden_state = output[0]
        elif hasattr(output, "last_hidden_state"):
            last_hidden_state = output.last_hidden_state
        else:
            last_hidden_state = output

        cls_embedding = last_hidden_state[:, 0, :]

        return cls_embedding

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
