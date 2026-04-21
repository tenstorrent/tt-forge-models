# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ERNIE-RNA model loader implementation for masked language modeling on RNA sequences.

ERNIE-RNA uses a BERT-like architecture with sinusoidal position embeddings.
The multimolecule package that provides the native ErnieRna classes is
incompatible with the pinned transformers version, so we load the model as
BertForMaskedLM with weight-key remapping.
"""
import importlib.util
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import BertConfig, BertForMaskedLM

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _load_rna_tokenizer_class():
    mod_key = "multimolecule.tokenisers.rna.tokenization_rna"
    if mod_key in sys.modules:
        return sys.modules[mod_key].RnaTokenizer

    existing_mm = sys.modules.get("multimolecule")
    if existing_mm and hasattr(existing_mm, "__spec__") and existing_mm.__spec__:
        site = str(Path(existing_mm.__spec__.origin).parent / "tokenisers")
    else:
        import sysconfig

        site_packages = sysconfig.get_path("purelib")
        site = str(Path(site_packages) / "multimolecule" / "tokenisers")

    for name in [
        "multimolecule",
        "multimolecule.tokenisers",
        "multimolecule.tokenisers.rna",
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    for mod_name, rel in [
        ("multimolecule.tokenisers.alphabet", "alphabet.py"),
        ("multimolecule.tokenisers.utils", "utils.py"),
        ("multimolecule.tokenisers.tokenization_utils", "tokenization_utils.py"),
        ("multimolecule.tokenisers.rna.utils", "rna/utils.py"),
        (mod_key, "rna/tokenization_rna.py"),
    ]:
        if mod_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(mod_name, f"{site}/{rel}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)

    return sys.modules[mod_key].RnaTokenizer


def _remap_ernierna_to_bert(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = "bert." + nk[len("model.") :]
        nk = nk.replace("layer_norm", "LayerNorm")

        if nk.startswith("lm_head.transform."):
            nk = nk.replace("lm_head.transform.", "cls.predictions.transform.")
        elif nk == "lm_head.bias":
            nk = "cls.predictions.bias"

        if "pairwise_bias_proj" in nk:
            continue

        new_sd[nk] = v
    return new_sd


class ModelVariant(StrEnum):
    ERNIERNA = "multimolecule/ernierna"


class ModelLoader(ForgeModel):
    """ERNIE-RNA model loader implementation for masked language modeling on RNA sequences."""

    _VARIANTS = {
        ModelVariant.ERNIERNA: ModelConfig(
            pretrained_model_name="multimolecule/ernierna",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ERNIERNA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ERNIE-RNA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        RnaTokenizer = _load_rna_tokenizer_class()
        self.tokenizer = RnaTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        repo = self._variant_config.pretrained_model_name

        config = BertConfig(
            vocab_size=26,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=1026,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
        )
        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = BertForMaskedLM(config)

        weights_path = hf_hub_download(repo, "model.safetensors")
        raw_sd = load_file(weights_path)
        bert_sd = _remap_ernierna_to_bert(raw_sd)
        model.load_state_dict(bert_sd, strict=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        masked_sequence = "uagc<mask>uaucagacugauguuga"

        inputs = self.tokenizer(
            masked_sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens
