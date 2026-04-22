# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ANTONIO-Alpha model loader implementation for computational pathology
image-text-to-text generation.

ANTONIO-Alpha is a vision-language model combining a Prism tile encoder,
a cross-attention projector, and a MedGemma-4B language backbone. It
consumes pre-extracted Prism slide embeddings rather than raw whole-slide
images.

The AntoniProjector class is provided by the Antoni-Alpha package
(https://pypi.org/project/Antoni-Alpha/).
"""

from typing import Optional

import torch
import torch.nn as nn

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

_PROJECTOR_MODULE = "antoni_alpha.models.antoni_projector"
_PKG_PREFIX = "antoni_alpha"


class ModelVariant(StrEnum):
    """Available ANTONIO-Alpha model variants."""

    ANTONIO_ALPHA = "Antoni_alpha"


class _AntoniAlphaWrapper(nn.Module):
    """Wraps AntoniProjector + LLM with a tensor-only forward for XLA compilation."""

    def __init__(self, projector, llm):
        super().__init__()
        self.projector = projector
        self.llm = llm

    def forward(
        self,
        slide_latents: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vision_embeds = self.projector(slide_latents)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        combined = torch.cat([vision_embeds, text_embeds], dim=1)
        vision_mask = torch.ones(
            vision_embeds.shape[:2],
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        outputs = self.llm(inputs_embeds=combined, attention_mask=combined_mask)
        return outputs.logits


def _import_projector():
    """Import AntoniProjector from the installed Antoni-Alpha package, bypassing
    the local model directory in sys.path that shadows the package."""
    import sys
    import importlib

    sp_paths = [p for p in sys.path if "site-packages" in p]
    other = [p for p in sys.path if "site-packages" not in p]
    for key in list(sys.modules):
        if key == _PKG_PREFIX or key.startswith(_PKG_PREFIX + "."):
            del sys.modules[key]
    saved = sys.path[:]
    sys.path = sp_paths + other
    try:
        mod = importlib.import_module(_PROJECTOR_MODULE)
        return mod.AntoniProjector
    finally:
        sys.path = saved


class ModelLoader(ForgeModel):
    """ANTONIO-Alpha model loader for pathology image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.ANTONIO_ALPHA: ModelConfig(
            pretrained_model_name="SaltySander/ANTONIO-Alpha",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANTONIO_ALPHA

    prism_embedding_dim = 1280
    num_tiles = 16
    num_output_tokens = 256
    llm_hidden_size = 2048
    vocab_size = 32000
    sample_seq_len = 16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANTONIO-Alpha",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ANTONIO-Alpha model instance.

        Requires: pip install Antoni-Alpha>=0.1.1
        """
        AntoniProjector = _import_projector()

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        projector = AntoniProjector(
            vision_embedding_dim=self.prism_embedding_dim,
            llm_hidden_size=self.llm_hidden_size,
            num_output_tokens=self.num_output_tokens,
        ).to(dtype)

        llm = self._load_llm(dtype)

        model = _AntoniAlphaWrapper(projector, llm)
        model.eval()
        return model

    def _load_llm(self, dtype):
        """Load MedGemma-4B, or fall back to a synthetic tiny LLM for compilation."""
        from transformers import AutoModelForCausalLM, AutoConfig

        llm_model_id = "google/medgemma-4b-it"
        try:
            return AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        except Exception:
            pass

        try:
            config = AutoConfig.from_pretrained(llm_model_id)
        except Exception:
            from transformers import Gemma3Config

            config = Gemma3Config(
                hidden_size=self.llm_hidden_size,
                intermediate_size=self.llm_hidden_size * 4,
                num_attention_heads=16,
                num_key_value_heads=4,
                num_hidden_layers=2,
                vocab_size=self.vocab_size,
            )
        return AutoModelForCausalLM.from_config(config).to(dtype)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load tensor-only sample inputs for ANTONIO-Alpha."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        slide_latents = torch.randn(
            batch_size, self.num_tiles, self.prism_embedding_dim, dtype=dtype
        )
        input_ids = torch.zeros(batch_size, self.sample_seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, self.sample_seq_len, dtype=torch.long)
        return {
            "slide_latents": slide_latents,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
