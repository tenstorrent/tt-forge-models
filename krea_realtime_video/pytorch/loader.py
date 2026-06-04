# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Krea Realtime Video 14B per-component loader for tt_forge_models.

krea/krea-realtime-video is a text-to-video ``WanModularPipeline``. Each
loadable component is exposed as its own ``ModelVariant`` so component tests
under ``tests/torch/models/krea_realtime/`` can request exactly one of them:

  - TEXT_ENCODER → UMT5EncoderModel    ~5.68B    (reused from Wan 2.1 14B)
  - TRANSFORMER  → CausalWanWrapper     ~14.29B   (CausalWanModel, from krea)
  - VAE          → VAEDecoderWrapper    ~0.13B    (AutoencoderKLWan, from Wan)

The pipeline reuses text_encoder/vae/tokenizer/scheduler from
``Wan-AI/Wan2.1-T2V-14B-Diffusers``; only the transformer weights come from
krea (see krea/krea-realtime-video/modular_model_index.json).

weight_fit (n150=12GiB, p150=32GiB, 85% budget):
  - transformer  → bf16 26.61GiB fits p150 weight-budget by ~0.6GiB only;
    video-DiT activations leave no single-chip headroom -> tensor_parallel.
  - text_encoder → bf16 10.58GiB exceeds n150 (10.2GiB), fits p150; sharded
    onto the uniform pipeline mesh alongside the transformer -> tensor_parallel.
  - vae          → fits both archs -> single_device (replicate-only on a mesh).

All I/O shapes/dtypes were captured from one real CPU forward per component;
see .claude/bringup/krea_realtime_video/io_spec.json.
"""

from typing import Any, Optional

import torch

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
from .src.model_utils import (
    DTYPE,
    KREA_REPO_ID,
    LATENT_H,
    LATENT_W,
    MAX_SEQ_LEN,
    MESH_NAMES,
    MESH_SHAPES,
    NUM_CHANNELS_LATENTS,
    NUM_FRAMES_PER_BLOCK,
    NUM_LATENT_FRAMES,
    TEXT_EMBED_DIM,
    UMT5_VOCAB_SIZE,
    WAN_REPO_ID,
    CausalWanWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_text_encoder_specs,
    shard_transformer_specs,
)

# Embedded per-component I/O spec (captured on CPU, bf16). Self-contained so
# the loader is reproducible without re-running capture.
COMPONENT_IO_SPEC = {
    "text_encoder": {
        "inputs": {
            "input_ids": {"shape": [1, MAX_SEQ_LEN], "dtype": "torch.int64"},
            "attention_mask": {"shape": [1, MAX_SEQ_LEN], "dtype": "torch.int64"},
        },
        "output": {
            "shape": [1, MAX_SEQ_LEN, TEXT_EMBED_DIM],
            "dtype": "torch.bfloat16",
        },
        "called_per_step": False,
    },
    "transformer": {
        "inputs": {
            "x": {
                "shape": [
                    1,
                    NUM_CHANNELS_LATENTS,
                    NUM_LATENT_FRAMES,
                    LATENT_H,
                    LATENT_W,
                ],
                "dtype": "torch.bfloat16",
            },
            "t": {"shape": [1, NUM_FRAMES_PER_BLOCK], "dtype": "torch.float32"},
            "context": {
                "shape": [1, MAX_SEQ_LEN, TEXT_EMBED_DIM],
                "dtype": "torch.bfloat16",
            },
        },
        "output": {
            "shape": [1, NUM_CHANNELS_LATENTS, NUM_LATENT_FRAMES, LATENT_H, LATENT_W],
            "dtype": "torch.bfloat16",
        },
        "called_per_step": True,
    },
    "vae": {
        "inputs": {
            "z": {
                "shape": [
                    1,
                    NUM_CHANNELS_LATENTS,
                    NUM_LATENT_FRAMES,
                    LATENT_H,
                    LATENT_W,
                ],
                "dtype": "torch.bfloat16",
            },
        },
        "output": {"shape": [1, 3, 9, 480, 832], "dtype": "torch.bfloat16"},
        "called_per_step": False,
    },
}


class ModelVariant(StrEnum):
    """Loadable components of the Krea Realtime Video 14B pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


# Map each variant to (HF repo, model_index subfolder).
_SOURCE = {
    ModelVariant.TEXT_ENCODER: (WAN_REPO_ID, "text_encoder"),
    ModelVariant.TRANSFORMER: (KREA_REPO_ID, "transformer"),
    ModelVariant.VAE: (WAN_REPO_ID, "vae"),
}


class ModelLoader(ForgeModel):
    """Load individual Krea Realtime Video components without holding the full pipeline.

    load_model() returns ONLY the requested component; load_inputs() builds
    synthetic tensors matched to that component's captured forward signature.
    """

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=WAN_REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=KREA_REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=WAN_REPO_ID),
    }

    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.component_name = _SOURCE[self._variant][1]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.MM_VIDEO_TTT
        )
        return ModelInfo(
            model="KreaRealtimeVideo",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return ONLY the requested component as a torch.nn.Module.

        TEXT_ENCODER → UMT5EncoderModel
        TRANSFORMER  → CausalWanWrapper  (raw CausalWanModel with simplified forward)
        VAE          → VAEDecoderWrapper (decoder-only, returns a plain tensor)
        """
        repo = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(repo, dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return CausalWanWrapper(load_transformer(repo, dtype))
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(repo, dtype))
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Synthetic inputs matched to the active component's captured signature.

        TEXT_ENCODER → [input_ids (1,512) int64, attention_mask (1,512) int64]
        TRANSFORMER  → [x (1,16,3,60,104) bf16, t (1,3) f32, context (1,512,4096) bf16]
        VAE          → [z (1,16,3,60,104) bf16]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, UMT5_VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.long
            )
            attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            x = torch.randn(
                1,
                NUM_CHANNELS_LATENTS,
                NUM_LATENT_FRAMES,
                LATENT_H,
                LATENT_W,
                dtype=dtype,
            )
            t = torch.full((1, NUM_FRAMES_PER_BLOCK), 1000.0, dtype=torch.float32)
            context = torch.randn(1, MAX_SEQ_LEN, TEXT_EMBED_DIM, dtype=dtype)
            return [x, t, context]

        if self._variant == ModelVariant.VAE:
            z = torch.randn(
                1,
                NUM_CHANNELS_LATENTS,
                NUM_LATENT_FRAMES,
                LATENT_H,
                LATENT_W,
                dtype=dtype,
            )
            return [z]

        raise ValueError(f"Unknown variant: {self._variant}")

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        """Extract the comparison tensor from a component's forward output."""
        if torch.is_tensor(output):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    # ------------------------------------------------------------------
    # Multichip TP (PROMOTION-ONLY). Refined by /model-bringup-multichip.
    # ------------------------------------------------------------------

    def get_mesh_config(self, num_devices: int):
        """Return ((batch, model) mesh shape, mesh names) for the active component.

        VAE is single_device -> (1, 1) on any device count (replicate-only).
        Supported device counts for TP components: 1, 2, 4, 8, 32.
        """
        if self._variant == ModelVariant.VAE:
            return (1, 1), MESH_NAMES
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict for the active component.

        Expects the same object returned by load_model():
          TEXT_ENCODER → UMT5EncoderModel
          TRANSFORMER  → CausalWanWrapper  (specs built from .transformer)
          VAE          → None (single_device, replicate-only)
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.VAE:
            return None
        raise ValueError(f"Unknown variant: {self._variant}")
