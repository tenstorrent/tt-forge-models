# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mochi GGUF model loader implementation.

Mochi is a ~10B text-to-video diffusion model by Genmo. This loader uses the
GGUF-quantized transformer repackaged by calcuis for ComfyUI / gguf-node.
Weights are dequantized via the gguf library's native dequantize function and
loaded into a standard MochiTransformer3DModel for XLA tracing.

Repository:
- https://huggingface.co/calcuis/mochi
"""

from typing import Optional

import torch
from diffusers import MochiTransformer3DModel

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

REPO_ID = "calcuis/mochi"


class ModelVariant(StrEnum):
    """Available Mochi GGUF model variants."""

    Q3_K_M = "Q3_K_M"


class ModelLoader(ForgeModel):
    """Mochi GGUF model loader for text-to-video generation."""

    _VARIANTS = {
        ModelVariant.Q3_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q3_K_M: "mochi-q3_k_m.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q3_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Mochi GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Mochi transformer, dequantizing GGUF weights to float."""
        try:
            import gguf as gguf_lib
            from diffusers.loaders.single_file_utils import (
                convert_mochi_transformer_checkpoint_to_diffusers,
            )
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "Install `gguf>=0.10.0` and `huggingface_hub` to load GGUF checkpoints."
            ) from e

        gguf_file = self._GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=REPO_ID, filename=gguf_file)
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        float_types = {
            gguf_lib.GGMLQuantizationType.F32,
            gguf_lib.GGMLQuantizationType.F16,
            gguf_lib.GGMLQuantizationType.BF16,
        }

        reader = gguf_lib.GGUFReader(gguf_path)
        checkpoint = {}
        for tensor in reader.tensors:
            if tensor.tensor_type in float_types:
                data = torch.from_numpy(tensor.data.copy())
            else:
                deq = gguf_lib.dequantize(tensor.data, tensor.tensor_type)
                data = torch.from_numpy(deq.copy()).to(torch.float32)
            checkpoint[tensor.name] = data

        diffusers_checkpoint = convert_mochi_transformer_checkpoint_to_diffusers(
            checkpoint
        )

        self.transformer = MochiTransformer3DModel()
        self.transformer.load_state_dict(diffusers_checkpoint, strict=True)
        self.transformer = self.transformer.to(dtype)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the Mochi transformer."""
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Mochi latent dimensions (12-channel latents, 6x temporal / 8x spatial
        # compression). Using small test dimensions.
        num_channels = config.in_channels
        num_frames = 2
        height = 12
        width = 12
        seq_len = 128  # matches MochiPipeline's max_sequence_length
        text_embed_dim = config.text_embed_dim

        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, height, width, dtype=dtype
        )

        timestep = torch.tensor([500], dtype=torch.long).expand(batch_size)

        encoder_hidden_states = torch.randn(
            batch_size, seq_len, text_embed_dim, dtype=dtype
        )

        encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
        }
