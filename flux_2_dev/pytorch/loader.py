# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev per-component loader for tt_forge_models.

black-forest-labs/FLUX.2-dev is a text-to-image DiffusionPipeline
(``Flux2Pipeline``). Each loadable component is exposed as its own
``ModelVariant`` so component tests under ``tests/torch/models/flux_2_dev/``
can request exactly one of them:

  - FLUX2_DEV_TEXT_ENCODER → Mistral3ForConditionalGeneration  ~24.0B  (bf16)
  - FLUX2_DEV_TRANSFORMER  → Flux2Transformer2DModel           ~32.2B  (bf16)
  - FLUX2_DEV_VAE          → AutoencoderKLFlux2 (decoder)       ~0.084B (fp32)

weight_fit (n150=12GiB, p150=32GiB, 85% budget):
  - text_encoder / transformer are weight-bound on EVERY single chip
    (48 GB / 64 GB bf16) -> parallelism_mode=tensor_parallel (PROMOTION-ONLY).
  - vae fits on both archs -> parallelism_mode=single_device.

All I/O shapes/dtypes were captured from one real CPU pass at 64x64 / 2 steps
(max_sequence_length=512); see src/utils.py and
.claude/bringup/flux_2_dev/scaffold_pipeline.json.
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
from .src.utils import (
    VAEDecoderWrapper,
    get_mesh_config,
    load_text_encoder,
    load_text_encoder_inputs,
    load_transformer,
    load_transformer_inputs,
    load_vae,
    load_vae_decoder_inputs,
    shard_text_encoder_specs,
    shard_transformer_specs,
)

HF_REPO = "black-forest-labs/FLUX.2-dev"

# Embedded per-component I/O spec (captured 64x64, seq=512). Self-contained so
# the loader is reproducible without re-running capture.
COMPONENT_IO_SPEC = {
    "text_encoder": {
        "inputs": {
            "input_ids": {"shape": [1, 512], "dtype": "torch.int64"},
            "attention_mask": {"shape": [1, 512], "dtype": "torch.int64"},
            "output_hidden_states": True,
        },
        "output": {"hidden_states": "tuple[41] of [1, 512, 5120] bf16"},
        "called_per_step": False,
    },
    "transformer": {
        "inputs": {
            "hidden_states": {"shape": [1, 16, 128], "dtype": "torch.bfloat16"},
            "encoder_hidden_states": {
                "shape": [1, 512, 15360],
                "dtype": "torch.bfloat16",
            },
            "timestep": {"shape": [1], "dtype": "torch.bfloat16"},
            "img_ids": {"shape": [1, 16, 4], "dtype": "torch.int64"},
            "txt_ids": {"shape": [1, 512, 4], "dtype": "torch.int64"},
            "guidance": {"shape": [1], "dtype": "torch.float32"},
        },
        "output": {"shape": [1, 16, 128], "dtype": "torch.bfloat16"},
        "called_per_step": True,
    },
    "vae": {
        "inputs": {"latent": {"shape": [1, 32, 8, 8], "dtype": "torch.bfloat16"}},
        "output": {"shape": [1, 3, 64, 64], "dtype": "torch.bfloat16"},
        "called_per_step": False,
    },
}

# Per-component parallelism mode (from weight_fit.json). transformer/text_encoder
# are tensor_parallel (promotion-only); vae is single_device.
PARALLELISM_MODE = {
    "text_encoder": "tensor_parallel",
    "transformer": "tensor_parallel",
    "vae": "single_device",
}


class ModelVariant(StrEnum):
    """Loadable components of the FLUX.2-dev pipeline."""

    FLUX2_DEV_TEXT_ENCODER = "Dev-TextEncoder"
    FLUX2_DEV_TRANSFORMER = "Dev-Transformer"
    FLUX2_DEV_VAE = "Dev-Vae"


# Map each variant to its HF model_index.json subfolder.
_SUBFOLDER = {
    ModelVariant.FLUX2_DEV_TEXT_ENCODER: "text_encoder",
    ModelVariant.FLUX2_DEV_TRANSFORMER: "transformer",
    ModelVariant.FLUX2_DEV_VAE: "vae",
}


class ModelLoader(ForgeModel):
    """Load individual FLUX.2-dev components without holding the full pipeline.

    load_model() returns ONLY the requested component; load_inputs() builds
    synthetic tensors matched to that component's captured forward signature.
    """

    _VARIANTS = {
        ModelVariant.FLUX2_DEV_TEXT_ENCODER: ModelConfig(pretrained_model_name=HF_REPO),
        ModelVariant.FLUX2_DEV_TRANSFORMER: ModelConfig(pretrained_model_name=HF_REPO),
        ModelVariant.FLUX2_DEV_VAE: ModelConfig(pretrained_model_name=HF_REPO),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX2_DEV_TRANSFORMER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.component_name = _SUBFOLDER[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.FLUX2_DEV_TEXT_ENCODER
            else ModelTask.MM_IMAGE_TTT  # FIXME: text-to-image task when available
        )
        return ModelInfo(
            model="FLUX.2-dev",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return ONLY the requested component as a torch.nn.Module.

        VAE is wrapped so forward(z) returns the decoded pixel tensor directly.
        """
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._variant == ModelVariant.FLUX2_DEV_TEXT_ENCODER:
            return load_text_encoder(model_name, dtype)
        if self._variant == ModelVariant.FLUX2_DEV_TRANSFORMER:
            return load_transformer(model_name, dtype)
        if self._variant == ModelVariant.FLUX2_DEV_VAE:
            vae = load_vae(model_name, dtype)
            return VAEDecoderWrapper(vae)
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Synthetic inputs matched to the active component's captured signature.

        VAE          → list [latent]            (positional, for run_graph_test)
        TRANSFORMER  → dict of forward kwargs    (per-denoise-step inputs)
        TEXT_ENCODER → dict of forward kwargs
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        height = kwargs.get("height", 64)
        width = kwargs.get("width", 64)
        seq_len = kwargs.get("seq_len", 512)

        if self._variant == ModelVariant.FLUX2_DEV_VAE:
            return [load_vae_decoder_inputs(dtype, height=height, width=width)]
        if self._variant == ModelVariant.FLUX2_DEV_TRANSFORMER:
            return load_transformer_inputs(
                dtype, height=height, width=width, seq_len=seq_len
            )
        if self._variant == ModelVariant.FLUX2_DEV_TEXT_ENCODER:
            return load_text_encoder_inputs(dtype, seq_len=seq_len)
        raise ValueError(f"Unknown variant: {self._variant}")

    # ------------------------------------------------------------------
    # Multichip TP (PROMOTION-ONLY). Refined by /model-bringup-multichip.
    # ------------------------------------------------------------------

    def get_mesh_config(self, num_devices: int):
        """((batch, model) mesh shape, mesh names) for the active component."""
        return get_mesh_config(num_devices)

    def load_shard_spec(self, model):
        """tensor -> partition_spec dict for the active component.

        VAE is single_device -> None (replicate-only on any mesh).
        """
        if self._variant == ModelVariant.FLUX2_DEV_TRANSFORMER:
            return shard_transformer_specs(model)
        if self._variant == ModelVariant.FLUX2_DEV_TEXT_ENCODER:
            return shard_text_encoder_specs(model)
        return None

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        """Extract the comparison tensor from a component's forward output."""
        if torch.is_tensor(output):
            return output
        if hasattr(output, "sample"):
            return output.sample
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            # Mistral3: tuple of per-layer hidden states; compare the last.
            hs = output.hidden_states
            return hs[-1] if isinstance(hs, (list, tuple)) else hs
        if isinstance(output, (list, tuple)):
            return output[0]
        return output
