# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gaurav0369/cipher-rnn2 model loader.

The Hugging Face repository ships a single ``pytorch_model.bin`` dict containing
the model state_dict, the original training config, and the character/cipher
vocabularies. This loader rebuilds the custom RNN architecture from those
hyperparameters and loads the pretrained weights.
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.model_utils import CipherRNN


class ModelVariant(StrEnum):
    """Available Gaurav0369/cipher-rnn2 model variants."""

    CIPHER_RNN2 = "cipher-rnn2"


class ModelLoader(ForgeModel):
    """Loader for the Gaurav0369/cipher-rnn2 character-to-cipher RNN."""

    _CHECKPOINT_FILE = "pytorch_model.bin"
    _SAMPLE_TEXT = "the quick brown fox jumps over the lazy dog"

    _VARIANTS = {
        ModelVariant.CIPHER_RNN2: ModelConfig(
            pretrained_model_name="Gaurav0369/cipher-rnn2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CIPHER_RNN2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._checkpoint = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="cipher-rnn2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_checkpoint(self):
        if self._checkpoint is None:
            weight_path = hf_hub_download(
                repo_id=self._variant_config.pretrained_model_name,
                filename=self._CHECKPOINT_FILE,
            )
            self._checkpoint = torch.load(
                weight_path, map_location="cpu", weights_only=False
            )
        return self._checkpoint

    def load_model(self, *, dtype_override=None, **kwargs):
        checkpoint = self._load_checkpoint()
        model_cfg = checkpoint["config"]["model"]

        model = CipherRNN(
            vocab_size=len(checkpoint["char2idx"]),
            embedding_dim=model_cfg["embedding_dim"],
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            output_size=len(checkpoint["cipher2idx"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, text=None):
        checkpoint = self._load_checkpoint()
        char2idx = checkpoint["char2idx"]
        seq_len = checkpoint["config"]["data"]["seq_len"]

        sample = text if text is not None else self._SAMPLE_TEXT
        sample = sample[:seq_len].ljust(seq_len)
        ids = [char2idx.get(c, 0) for c in sample]

        input_ids = torch.tensor([ids], dtype=torch.long).repeat(batch_size, 1)
        return input_ids

    def decode_output(self, outputs, inputs=None):
        checkpoint = self._load_checkpoint()
        idx2cipher = {v: k for k, v in checkpoint["cipher2idx"].items()}

        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        predicted_ids = torch.argmax(logits, dim=-1)

        decoded = []
        for row in predicted_ids:
            decoded.append(" ".join(idx2cipher.get(int(i), "") for i in row))
        return decoded
