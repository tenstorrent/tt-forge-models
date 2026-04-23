# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ANTONI-Alpha model loader implementation for computational pathology
image-text-to-text generation.

ANTONI-Alpha is a vision-language model combining a Prism tile encoder,
a cross-attention projector, and a MedGemma-4B language backbone. It
consumes pre-extracted Prism slide embeddings rather than raw whole-slide
images.

The AntoniAlphaPreTrained class is provided by the antoni_alpha package
(https://github.com/computationalpathologygroup/ANTONI-Alpha).
"""

from typing import Optional

import torch

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


class ModelVariant(StrEnum):
    """Available ANTONI-Alpha model variants."""

    ANTONI_ALPHA = "antoni_alpha"


class ModelLoader(ForgeModel):
    """ANTONI-Alpha model loader for pathology image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.ANTONI_ALPHA: ModelConfig(
            pretrained_model_name="SaltySander/ANTONI-Alpha",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANTONI_ALPHA

    # Prism tile embedding dimension expected by the projector.
    prism_embedding_dim = 1280
    num_tiles = 16
    sample_prompt = "Describe this tissue."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANTONI-Alpha",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ANTONI-Alpha model instance.

        Requires the antoni_alpha package:
            pip install git+https://github.com/computationalpathologygroup/ANTONI-Alpha.git
        """
        import sys

        # The local 'Antoni_alpha/' project directory shadows the installed Antoni-Alpha
        # package in sys.path. Temporarily elevate site-packages to import from the
        # correct installed location.
        _site_pkgs = [p for p in sys.path if "site-packages" in p]
        _saved_path = sys.path[:]
        for _k in list(sys.modules.keys()):
            if _k == "antoni_alpha" or _k.startswith("antoni_alpha."):
                del sys.modules[_k]
        sys.path = _site_pkgs + [p for p in sys.path if p not in _site_pkgs]
        try:
            from antoni_alpha.models.antoni_pretrained import AntoniAlphaPreTrained
        finally:
            sys.path = _saved_path

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AntoniAlphaPreTrained.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ANTONI-Alpha model.

        ANTONI-Alpha operates on pre-extracted Prism slide embeddings of
        shape [batch_size, num_tiles, 1280], paired with an OpenAI-style
        conversation. Synthetic slide latents are produced here to match
        the expected feature dimensions.
        """
        slide_latents = torch.randn(
            batch_size, self.num_tiles, self.prism_embedding_dim
        )

        if dtype_override is not None:
            slide_latents = slide_latents.to(dtype_override)

        conversations = [[{"role": "user", "content": self.sample_prompt}]] * batch_size

        return {
            "slide_latents": slide_latents,
            "conversations": conversations,
        }
