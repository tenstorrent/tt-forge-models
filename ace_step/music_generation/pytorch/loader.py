# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 (turbo) music-generation DiT denoiser loader.

ACE-Step 1.5 is a text-to-music diffusion pipeline. The heavy per-step compute
is the Diffusion Transformer (DiT) denoiser ``AceStepDiTModel`` (``model.decoder``
inside ``AceStepConditionGenerationModel``). This is the *key* component of the
pipeline and the one brought up on device here; the text encoder, planner LM and
VAE are separate loaders (see the sibling ``ace_step`` task dirs).

The model ships as HuggingFace ``trust_remote_code`` custom code living in the
``acestep-v15-turbo`` subfolder of the repo. Two non-obvious facts handled below:

  * transformers >= 5 always constructs models under ``torch.device("meta")``;
    the third-party ``ResidualFSQ`` quantizer calls ``.item()`` during ``__init__``
    which crashes on meta tensors. We therefore construct the module on CPU
    ourselves and load the safetensors weights, instead of ``from_pretrained``.
  * The custom code is in a repo *subfolder*, but its ``auto_map`` points at the
    repo root, so ``from_pretrained(..., subfolder=...)`` cannot find it. We
    snapshot-download the subfolder and load the module from the local path.
"""
import os
from typing import Optional

import torch
from torch import nn

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
    """Available ACE-Step 1.5 DiT variants."""

    TURBO = "turbo"


class _DenoiserWrapper(nn.Module):
    """Thin wrapper exposing a single-tensor forward over the DiT decoder.

    The raw ``AceStepDiTModel`` returns ``(velocity, past_key_values)``; the tt
    model tester expects a single tensor, so we disable the cache and return the
    predicted flow/velocity tensor only.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
    ):
        out = self.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
        )
        return out[0]


def _build_full_model(dtype: torch.dtype):
    """Construct the full ACE-Step condition-generation model on CPU and load weights.

    Returns the ``AceStepConditionGenerationModel`` instance (eval, on CPU, cast to
    ``dtype``). Bypasses ``from_pretrained``'s meta-device init (see module docstring).
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors.torch import load_file

    repo = "ACE-Step/Ace-Step1.5"
    sub = os.path.join(
        snapshot_download(repo, allow_patterns=["acestep-v15-turbo/*"]),
        "acestep-v15-turbo",
    )
    config = AutoConfig.from_pretrained(sub, trust_remote_code=True)
    # eager attention compiles cleanly; flash-attn-2 (the repo default) is unavailable.
    config._attn_implementation = "eager"
    model_cls = get_class_from_dynamic_module(
        "modeling_acestep_v15_turbo.AceStepConditionGenerationModel", sub
    )
    with torch.device("cpu"):
        model = model_cls(config)
    state_dict = load_file(os.path.join(sub, "model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(dtype).eval()
    return model, config


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 turbo DiT denoiser loader (key component of the pipeline)."""

    _VARIANTS = {
        ModelVariant.TURBO: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TURBO

    # Denoiser input geometry (see modeling_acestep_v15_turbo.AceStepDiTModel.forward).
    # Native acoustic latent is 25 Hz; 10 s of audio -> 250 frames. We use a smaller
    # even sequence length here to keep the device compile tractable; the geometry
    # (channels / conditioning dims) matches the real pipeline exactly.
    BATCH_SIZE = 1
    SEQ_LEN = 128  # acoustic frames (must be even for patch_size=2)
    ENC_LEN = 64  # conditioning sequence length
    ACOUSTIC_DIM = 64  # audio_acoustic_hidden_dim (in/out latent channels)
    CONTEXT_DIM = 128  # src_latents (64) + chunk_masks (64)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._config = None
        self.hidden_size = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ACE-Step 1.5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        """Return the DiT denoiser wrapped for single-tensor output.

        Args:
            dtype_override: Optional torch.dtype. Defaults to bfloat16 (model native).

        Returns:
            torch.nn.Module: ``_DenoiserWrapper`` around ``AceStepDiTModel``.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model, config = _build_full_model(dtype)
        self._config = config
        self.hidden_size = config.hidden_size
        return _DenoiserWrapper(model.decoder)

    def load_inputs(self, dtype_override=None, **kwargs):
        """Return deterministic synthetic denoiser inputs at the pipeline geometry.

        Args:
            dtype_override: Optional torch.dtype for the float tensors.

        Returns:
            dict: kwargs matching ``_DenoiserWrapper.forward``.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        # hidden_size must be known to size encoder_hidden_states; load config if needed.
        if self.hidden_size is None:
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig

            sub = os.path.join(
                snapshot_download(
                    "ACE-Step/Ace-Step1.5", allow_patterns=["acestep-v15-turbo/*"]
                ),
                "acestep-v15-turbo",
            )
            self.hidden_size = AutoConfig.from_pretrained(
                sub, trust_remote_code=True
            ).hidden_size

        b, t, enc = self.BATCH_SIZE, self.SEQ_LEN, self.ENC_LEN
        gen = torch.Generator().manual_seed(0)
        return {
            "hidden_states": torch.randn(
                b, t, self.ACOUSTIC_DIM, generator=gen, dtype=torch.float32
            ).to(dtype),
            "timestep": torch.full((b,), 0.5, dtype=dtype),
            "timestep_r": torch.full((b,), 0.5, dtype=dtype),
            "attention_mask": torch.ones(b, t, dtype=dtype),
            "encoder_hidden_states": torch.randn(
                b, enc, self.hidden_size, generator=gen, dtype=torch.float32
            ).to(dtype),
            "encoder_attention_mask": torch.ones(b, enc, dtype=dtype),
            "context_latents": torch.randn(
                b, t, self.CONTEXT_DIM, generator=gen, dtype=torch.float32
            ).to(dtype),
        }
