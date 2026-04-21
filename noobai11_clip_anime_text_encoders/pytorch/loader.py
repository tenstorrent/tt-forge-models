# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NoobAI 1.1 anime text encoder loader implementation.

The Anzhc/Noobai11-CLIP-L-and-BigG-Anime-Text-Encoders repository ships two
standalone SDXL text encoder weight files that are meant to replace the
text_encoder / text_encoder_2 components of a noobai-XL 1.0 pipeline:

- "Anzhc Noobai11 CLIP L Anime.safetensors": CLIPTextModel weights derived
  from openai/clip-vit-large-patch14 and used as SDXL text_encoder.
- "Bluvoll Noobai11 CLIP G Anime.safetensors": CLIPTextModelWithProjection
  weights derived from laion/CLIP-ViT-bigG-14-laion2B-39B-b160k and used as
  SDXL text_encoder_2.

Available variants:
- CLIP_L: CLIPTextModel with the anime-tuned CLIP-L weights.
- CLIP_G: CLIPTextModelWithProjection with the anime-tuned CLIP-G weights.
"""

from typing import Optional

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

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


REPO_ID = "Anzhc/Noobai11-CLIP-L-and-BigG-Anime-Text-Encoders"
BASE_SDXL_REPO = "Laxhar/noobai-XL-1.0"

CLIP_L_FILENAME = "Anzhc Noobai11 CLIP L Anime.safetensors"
CLIP_G_FILENAME = "Bluvoll Noobai11 CLIP G Anime.safetensors"


class ModelVariant(StrEnum):
    """Available NoobAI 1.1 anime text encoder variants."""

    CLIP_L = "clip_l"
    CLIP_G = "clip_g"


class ModelLoader(ForgeModel):
    """NoobAI 1.1 anime text encoder loader implementation."""

    _VARIANTS = {
        ModelVariant.CLIP_L: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.CLIP_G: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.CLIP_L

    sample_prompt = "masterpiece, best quality, 1girl, blue hair, detailed eyes"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Noobai11 CLIP Anime Text Encoders",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _tokenizer_subfolder(self) -> str:
        return "tokenizer" if self._variant == ModelVariant.CLIP_L else "tokenizer_2"

    def _text_encoder_subfolder(self) -> str:
        return (
            "text_encoder" if self._variant == ModelVariant.CLIP_L else "text_encoder_2"
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the selected NoobAI 1.1 anime text encoder.

        Args:
            dtype_override: Optional torch.dtype to cast the model to.

        Returns:
            torch.nn.Module: CLIPTextModel for the CLIP-L variant or
            CLIPTextModelWithProjection for the CLIP-G variant, with the
            anime-tuned weights loaded in.
        """
        subfolder = self._text_encoder_subfolder()

        config = CLIPTextConfig.from_pretrained(BASE_SDXL_REPO, subfolder=subfolder)
        if self._variant == ModelVariant.CLIP_L:
            model = CLIPTextModel(config)
            weights_filename = CLIP_L_FILENAME
        else:
            model = CLIPTextModelWithProjection(config)
            weights_filename = CLIP_G_FILENAME

        weights_path = hf_hub_download(REPO_ID, weights_filename)
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return tokenized sample inputs for the text encoder.

        Args:
            dtype_override: Unused; tokenizer output is integer input_ids.
            batch_size: Number of prompts to batch.

        Returns:
            dict: Input tensors with input_ids suitable for the text encoder.
        """
        if self.tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                BASE_SDXL_REPO, subfolder=self._tokenizer_subfolder()
            )

        prompts = [self.sample_prompt] * batch_size
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {"input_ids": inputs["input_ids"]}
