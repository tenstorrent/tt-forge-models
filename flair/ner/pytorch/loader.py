# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair NER model loader implementation for multilingual named entity recognition.
"""

from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Flair NER model variants."""

    NER_MULTI_FAST = "ner-multi-fast"


class ModelLoader(ForgeModel):
    """Flair NER model loader implementation for multilingual named entity recognition."""

    _VARIANTS = {
        ModelVariant.NER_MULTI_FAST: ModelConfig(
            pretrained_model_name="flair/ner-multi-fast",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_MULTI_FAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "George Washington went to Washington"

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "ner-multi-fast"
        return ModelInfo(
            model="Flair",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _import_flair_module(submodule):
        import importlib
        import site
        import sys

        site_dirs = site.getsitepackages()
        saved = sys.path[:]
        for d in reversed(site_dirs):
            sys.path.insert(0, d)
        try:
            for key in list(sys.modules):
                if key == "flair" or key.startswith("flair."):
                    del sys.modules[key]
            return importlib.import_module(submodule)
        finally:
            sys.path[:] = saved

    @staticmethod
    def _patch_lstm_flat_weights():
        import weakref

        import torch.nn as nn

        if getattr(nn.RNNBase, "_flair_patched", False):
            return

        def _patched_setstate(self, d):
            nn.Module.__setstate__(self, d)
            if "all_weights" in d:
                self._all_weights = d["all_weights"]
            if "proj_size" not in d:
                self.proj_size = 0
            if not isinstance(self._all_weights[0][0], str):
                num_layers = self.num_layers
                num_directions = 2 if self.bidirectional else 1
                self._flat_weights_names = []
                self._all_weights = []
                for layer in range(num_layers):
                    for direction in range(num_directions):
                        suffix = "_reverse" if direction == 1 else ""
                        weights = [
                            f"weight_ih_l{layer}{suffix}",
                            f"weight_hh_l{layer}{suffix}",
                            f"bias_ih_l{layer}{suffix}",
                            f"bias_hh_l{layer}{suffix}",
                            f"weight_hr_l{layer}{suffix}",
                        ]
                        if self.bias:
                            n = 5 if self.proj_size > 0 else 4
                        else:
                            n = 3 if self.proj_size > 0 else 2
                        weights = weights[:n]
                        self._all_weights.append(weights)
                        self._flat_weights_names.extend(weights)
            else:
                self._flat_weights_names = [w for ws in self._all_weights for w in ws]
            self._flat_weights = [
                getattr(self, wn) if hasattr(self, wn) else None
                for wn in self._flat_weights_names
            ]
            self._flat_weight_refs = [
                weakref.ref(w) if w is not None else None for w in self._flat_weights
            ]

        nn.RNNBase.__setstate__ = _patched_setstate
        nn.RNNBase._flair_patched = True

    def load_model(self, **kwargs):
        import torch
        import torch.nn as nn

        self._patch_lstm_flat_weights()
        flair_models = self._import_flair_module("flair.models")
        tagger = flair_models.SequenceTagger.load(self.model_name)

        self.embedding_dim = tagger.embeddings.embedding_length
        self.seq_len = 8

        class FlairNERWrapper(nn.Module):
            def __init__(self, tagger):
                super().__init__()
                self.embedding2nn = tagger.embedding2nn
                self.rnn = tagger.rnn
                self.linear = tagger.linear

            def forward(self, sentence_tensor):
                x = self.embedding2nn(sentence_tensor)
                x, _ = self.rnn(x)
                return self.linear(x)

        model = FlairNERWrapper(tagger)
        model.eval()
        return model

    def load_inputs(self, **kwargs):
        import torch

        sentence_tensor = torch.randn(1, self.seq_len, self.embedding_dim)
        return (sentence_tensor,)
