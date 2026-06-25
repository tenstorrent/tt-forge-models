# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pyramid Flow SD3 text-encoder loaders.

The SD3 variant of Pyramid Flow (``rain1011/pyramid-flow-sd3``) conditions the
denoiser on three standard SD3 text encoders (no vendored code needed — they are
plain HuggingFace classes loaded straight from the model repo's subfolders):

  * ``clip_l``  — ``CLIPTextModelWithProjection`` (text_encoder,  hidden 768)
  * ``clip_g``  — ``CLIPTextModelWithProjection`` (text_encoder_2, hidden 1280)
  * ``t5_xxl``  — ``T5EncoderModel``             (text_encoder_3, d_model 4096)

The denoiser consumes the T5 last-hidden-state as ``encoder_hidden_states``
(joint_attention_dim=4096) and the concatenated CLIP pooled projections
(768+1280=2048) as ``pooled_projections``. Each encoder is an independently
compilable single-forward component, exposed here one per variant.
"""

from typing import Optional

import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

_HF_REPO = "rain1011/pyramid-flow-sd3"


class ModelVariant(StrEnum):
    """Available Pyramid Flow SD3 text-encoder components."""

    CLIP_L = "clip_l"
    CLIP_G = "clip_g"
    T5_XXL = "t5_xxl"


# variant -> (HF subfolder, transformers class name, max token length)
_SUBFOLDER = {
    ModelVariant.CLIP_L: ("text_encoder", "CLIPTextModelWithProjection", 77),
    ModelVariant.CLIP_G: ("text_encoder_2", "CLIPTextModelWithProjection", 77),
    ModelVariant.T5_XXL: ("text_encoder_3", "T5EncoderModel", 128),
}

# tokenizer subfolders (parallel to the encoder subfolders)
_TOKENIZER_SUBFOLDER = {
    ModelVariant.CLIP_L: "tokenizer",
    ModelVariant.CLIP_G: "tokenizer_2",
    ModelVariant.T5_XXL: "tokenizer_3",
}


class ModelLoader(ForgeModel):
    """Loader for the three Pyramid Flow SD3 text encoders (one per variant)."""

    _VARIANTS = {
        v: ModelConfig(pretrained_model_name=_HF_REPO) for v in ModelVariant
    }

    DEFAULT_VARIANT = ModelVariant.CLIP_L

    prompt = "A serene mountain lake at sunrise, cinematic, highly detailed"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PyramidFlow-SD3-TextEncoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            _HF_REPO, subfolder=_TOKENIZER_SUBFOLDER[self._variant]
        )
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        subfolder, cls_name, _ = _SUBFOLDER[self._variant]
        import transformers

        model_cls = getattr(transformers, cls_name)
        dtype = dtype_override if dtype_override is not None else torch.float32
        model = model_cls.from_pretrained(
            _HF_REPO, subfolder=subfolder, torch_dtype=dtype
        )
        return model.to(dtype=dtype).eval()

    def load_inputs(self, dtype_override=None, **kwargs):
        if self._tokenizer is None:
            self._load_tokenizer()
        _, _, max_len = _SUBFOLDER[self._variant]
        enc = self._tokenizer(
            self.prompt,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        # Text encoders take integer token ids / masks; no float dtype override.
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
