# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
e5-omni-7B model loader for omni-modal embedding generation.

Produces unified embeddings across text, image, audio, and video using a
Qwen2.5-Omni-7B backbone. This loader exercises the text-only path so the
harness input is a plain tokenized sentence.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available e5-omni model variants."""

    E5_OMNI_7B = "7B"


class ModelLoader(ForgeModel):
    """e5-omni-7B omni-modal embedding model loader."""

    _VARIANTS = {
        ModelVariant.E5_OMNI_7B: LLMModelConfig(
            pretrained_model_name="Haon-Chen/e5-omni-7B",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.E5_OMNI_7B

    sample_sentences = [
        "How to cook Mapo Tofu?",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="e5-omni-7B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
            Qwen2_5OmniThinkerConfig,
        )
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        # qwen2_5_omni_thinker is in transformers 5.2.0 but not registered in auto mappings
        AutoConfig.register(
            "qwen2_5_omni_thinker", Qwen2_5OmniThinkerConfig, exist_ok=True
        )
        AutoModel.register(
            Qwen2_5OmniThinkerConfig,
            Qwen2_5OmniThinkerForConditionalGeneration,
            exist_ok=True,
        )

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = getattr(self._variant_config, "max_length", 512)

        inputs = self.tokenizer(
            self.sample_sentences,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        """Extract and L2-normalize the embedding from the model output.

        e5-omni embeddings are the last hidden state at the last token,
        normalized along the embedding dim.
        """
        if hasattr(output, "last_hidden_state"):
            hidden_states = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            hidden_states = output[0]
        else:
            hidden_states = output

        emb = hidden_states[:, -1, :]
        return F.normalize(emb, p=2, dim=-1)
