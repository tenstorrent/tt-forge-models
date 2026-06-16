# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Infinity model loader implementation.
"""

from types import SimpleNamespace
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
from .src import model as _m
from .src.model_utils import build_forward_inputs


_HF_REPO_ID = "FoundationVision/Infinity"
_TRANSFORMER_FILENAME = "infinity_2b_reg.pth"
_VAE_FILENAME = "infinity_vae_d32reg.pth"
_TEXT_ENCODER_HF_ID = "google/flan-t5-xl"


class ModelVariant(StrEnum):
    """Available Infinity model variants."""

    INFINITY_2B = "2B"


class ModelLoader(ForgeModel):
    """Infinity 2B text-to-image model loader."""

    _VARIANTS = {
        ModelVariant.INFINITY_2B: ModelConfig(
            pretrained_model_name="FoundationVision/Infinity",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INFINITY_2B

    _PN = "1M"
    _TEXT_CHANNELS = 2048
    _VAE_TYPE = 32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Infinity",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def _build_run_args(self):
        """SimpleNamespace mirroring the args ``run_infinity`` loaders read."""
        return SimpleNamespace(
            model_type="infinity_2b",
            model_path=hf_hub_download(
                repo_id=_HF_REPO_ID, filename=_TRANSFORMER_FILENAME
            ),
            checkpoint_type="torch",
            enable_model_cache=0,
            pn=self._PN,
            use_bit_label=1,
            add_lvl_embeding_only_first_block=1,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            text_channels=self._TEXT_CHANNELS,
            apply_spatial_patchify=0,
            use_flex_attn=0,
            bf16=0,
            vae_type=self._VAE_TYPE,
            vae_path=hf_hub_download(repo_id=_HF_REPO_ID, filename=_VAE_FILENAME),
            text_encoder_ckpt=_TEXT_ENCODER_HF_ID,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Infinity 2B transformer.

        Side effects: also loads the T5-XL tokenizer/encoder and the BSQ-VAE
        and stores them on ``self`` so ``load_inputs`` can build realistic
        conditioning tensors.

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the transformer
                weights to (e.g. ``torch.bfloat16`` for TT execution).

        Returns:
            torch.nn.Module: The Infinity transformer.
        """
        run_args = self._build_run_args()

        self.tokenizer, self.text_encoder = _m.load_tokenizer(
            t5_path=run_args.text_encoder_ckpt
        )
        self.vae = _m.load_visual_tokenizer(run_args)
        self.model = _m.load_transformer(self.vae, run_args)
        self.model.eval()

        if dtype_override is not None:
            self.model = self.model.to(dtype_override)

        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1, prompt=None):
        """Return a single forward-pass input dict for the Infinity transformer.

        Delegates input construction to
        :func:`.src.model_utils.build_forward_inputs`. Targets the
        training-style ``forward`` path (``cfg_infer=False``) -- a single
        traceable pass that returns logits, not a sampling loop.

        Args:
            dtype_override: Optional ``torch.dtype`` for the tensor inputs.
            batch_size: Replication factor for the prompt.
            prompt: Optional override prompt; defaults to a fixed string.

        Returns:
            list: positional args for ``Infinity.forward`` --
                ``[label_B_or_BLT, x_BLC_wo_prefix, scale_schedule]`` where
                ``label_B_or_BLT`` is ``(kv_compact, lens, cu_seqlens_k,
                max_seqlen_k)``.
        """
        if self.text_encoder is None:
            raise RuntimeError("load_model() must be called before load_inputs().")
        return build_forward_inputs(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            pn=self._PN,
            batch_size=batch_size,
            prompt=prompt,
            dtype_override=dtype_override,
        )
