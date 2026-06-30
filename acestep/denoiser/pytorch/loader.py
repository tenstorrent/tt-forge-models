# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 DiT denoiser loader implementation.

ACE-Step 1.5 (ACE-Step/Ace-Step1.5) is a text-to-music diffusion pipeline. This
loader brings up its key component: the ``AceStepDiTModel`` diffusion transformer
("acestep-v15-turbo"), the per-step denoising compute of the pipeline.

The checkpoint ships as custom code (``trust_remote_code``) whose top-level wrapper
``AceStepConditionGenerationModel`` constructs a ``ResidualFSQ`` quantizer that calls
``.item()`` in ``__init__`` -- incompatible with transformers >= 5 ``from_pretrained``,
which builds every model under ``with torch.device("meta")``. The FSQ quantizer lives
only in the (cover-song) audio tokenizer, not in the denoiser, so we build the
``AceStepDiTModel`` directly on CPU and load only the ``decoder.*`` weights from the
checkpoint. See requirements.txt for the (einops / vector_quantize_pytorch) deps the
custom module imports at import time.
"""

import importlib
import os
import sys
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig

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


class ModelVariant(StrEnum):
    """Available ACE-Step 1.5 denoiser variants."""

    V15_TURBO = "v15-turbo"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 DiT denoiser loader implementation."""

    _VARIANTS = {
        ModelVariant.V15_TURBO: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V15_TURBO

    # Sub-directory of the repo holding the turbo denoiser custom code + weights.
    _SUBFOLDER = "acestep-v15-turbo"

    # Representative latent shapes for a single denoise step. Audio has no fixed
    # "native resolution"; the pipeline denoises in chunks. A 512-frame latent at the
    # VAE's 25 Hz latent rate (48 kHz / product(downsampling_ratios)=1920) is ~20.5 s.
    seq_len = 512  # latent frames (must be even: proj_in is Conv1d stride=patch_size=2)
    encoder_seq_len = 128  # packed conditioning length (lyric + timbre + text)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="ACE-Step 1.5 DiT denoiser",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _resolve_subdir(self):
        """Download (cached) the turbo sub-folder and return its local path."""
        local = snapshot_download(
            self._variant_config.pretrained_model_name,
            allow_patterns=[f"{self._SUBFOLDER}/*"],
        )
        return os.path.join(local, self._SUBFOLDER)

    def load_model(self, dtype_override=None, **kwargs):
        """Build the AceStepDiTModel denoiser and load its weights.

        Args:
            dtype_override: Optional torch.dtype. Defaults to bfloat16.

        Returns:
            torch.nn.Module: the DiT denoiser (AceStepDiTModel).
        """
        dtype = dtype_override or torch.bfloat16
        sub = self._resolve_subdir()

        # Import the custom modeling module directly from the snapshot. The module's
        # relative import falls back to a top-level import, so adding `sub` to sys.path
        # is enough to resolve `configuration_acestep_v15`.
        if sub not in sys.path:
            sys.path.insert(0, sub)
        modeling = importlib.import_module("modeling_acestep_v15_turbo")

        config = AutoConfig.from_pretrained(sub, trust_remote_code=True)
        self._config = config

        # Build the DiT on CPU (avoids the transformers>=5 meta-device init that the
        # FSQ quantizer in the full wrapper cannot survive), then load decoder weights.
        model = modeling.AceStepDiTModel(config)
        state_dict = load_file(os.path.join(sub, "model.safetensors"))
        decoder_sd = {
            k[len("decoder.") :]: v
            for k, v in state_dict.items()
            if k.startswith("decoder.")
        }
        model.load_state_dict(decoder_sd, strict=True)

        model = model.to(dtype).eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build a single-denoise-step input batch for the DiT.

        The denoiser predicts a flow field from a noisy latent conditioned on
        timesteps, packed encoder conditioning, and context latents (source latents +
        chunk masks). Internal attention masks are recomputed inside the model, so the
        passed masks are placeholders.

        Returns:
            dict: keyword arguments for AceStepDiTModel.forward.
        """
        dtype = dtype_override or torch.bfloat16
        if self._config is None:
            self.load_model(dtype_override=dtype)
        cfg = self._config

        c_latent = cfg.audio_acoustic_hidden_dim  # noisy latent channels (64)
        hidden_size = cfg.hidden_size  # encoder conditioning width (2048)
        b, t, lenc = batch_size, self.seq_len, self.encoder_seq_len

        torch.manual_seed(0)
        inputs = {
            # Noisy acoustic latent x_t: [B, T, 64]
            "hidden_states": torch.randn(b, t, c_latent, dtype=dtype),
            "timestep": torch.rand(b, dtype=dtype),
            "timestep_r": torch.rand(b, dtype=dtype),
            # Placeholder padding mask (model recomputes 4D masks internally).
            "attention_mask": torch.ones(b, t, dtype=dtype),
            # Packed conditioning embeddings: [B, L_enc, hidden_size]
            "encoder_hidden_states": torch.randn(b, lenc, hidden_size, dtype=dtype),
            "encoder_attention_mask": torch.ones(b, lenc, dtype=dtype),
            # Context latents = cat([src_latents(64), chunk_masks(64)]): [B, T, 128]
            "context_latents": torch.randn(b, t, 2 * c_latent, dtype=dtype),
            # Avoid building a KV cache object (keeps output a plain tensor tuple).
            "use_cache": False,
        }
        return inputs
