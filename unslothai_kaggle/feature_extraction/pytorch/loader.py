# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unslothai Kaggle model loader implementation for feature extraction.
"""
import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from typing import Optional

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


# unslothai/kaggle has num_attention_heads=0 so compute_default_rope_parameters
# raises ZeroDivisionError via the falsy `head_dim=0 or 0//0` expression.
# Patch the staticmethod to short-circuit for zero-attention-head configs.
_orig_compute_rope = LlamaRotaryEmbedding.compute_default_rope_parameters


@staticmethod  # type: ignore[misc]
def _safe_compute_rope_parameters(config, device=None, seq_len=None, **kwargs):
    if not getattr(config, "num_attention_heads", 1):
        return torch.zeros(0, dtype=torch.float32), 1.0
    return _orig_compute_rope(config, device=device, seq_len=seq_len, **kwargs)


LlamaRotaryEmbedding.compute_default_rope_parameters = _safe_compute_rope_parameters


class ModelVariant(StrEnum):
    """Available Unslothai Kaggle model variants for feature extraction."""

    UNSLOTHAI_KAGGLE = "unslothai/kaggle"


class ModelLoader(ForgeModel):
    """Unslothai Kaggle model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.UNSLOTHAI_KAGGLE: ModelConfig(
            pretrained_model_name="unslothai/kaggle",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTHAI_KAGGLE

    sample_text = "This is an example sentence for feature extraction."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Unslothai Kaggle",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # The unslothai/kaggle config has all-zero dimensions (vocab_size=0,
        # hidden_size=0, num_attention_heads=0, num_hidden_layers=0).
        # LlamaConfig.__init__ in transformers>=4.40 computes head_dim as
        # hidden_size // num_attention_heads, raising ZeroDivisionError.
        # Inject head_dim=0 into the raw config dict to bypass the division.
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name)
        if config_dict.get("num_attention_heads", 0) == 0:
            config_dict.setdefault("head_dim", 0)
        model_type = config_dict.pop("model_type", "llama")
        config = AutoConfig.for_model(model_type, **config_dict)

        model = AutoModel.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        # unslothai/kaggle has vocab_size=0 and hidden_size=0; normal tokenized
        # inputs cause IndexError in the embedding layer.  Use an empty sequence
        # (length 0) so all tensor operations on empty tensors succeed.
        inputs = {"input_ids": torch.zeros(1, 0, dtype=torch.long)}
        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)
        return inputs
