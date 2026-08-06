# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 1.5 model loader implementation.

SD1.4 (``CompVis/stable-diffusion-v1-4``) and SD1.5
(``stable-diffusion-v1-5/stable-diffusion-v1-5``) share the same
``UNet2DConditionModel`` architecture, ``LMSDiscreteScheduler`` and CLIP
text encoder; they only differ in pretrained weights. We still ship them
as separate loader packages so each bringup can advance independently and
each can carry its own ModelInfo / dashboards / status in the test runner.

``load_model`` returns the SD1.5 UNet (an ``nn.Module``) — the format the
tt-xla model tester expects.
"""

from typing import Optional

import torch
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

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

# CLIP text encoder used by SD1.x (openai/clip-vit-large-patch14).
CLIP_REPO_ID = "openai/clip-vit-large-patch14"
# CLIP context length (tokenizer model_max_length) and vocab size.
CLIP_MAX_SEQ_LEN = 77
CLIP_VOCAB_SIZE = 49408

# (batch, model) mesh shapes by device count — matches the sibling image loaders.
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


class CLIPTextEncoderWrapper(torch.nn.Module):
    """Run ``CLIPTextModel`` as a stateless encoder returning a plain tensor.

    Pins ``return_dict=False`` so graph capture sees a pure tensor
    (last_hidden_state) rather than a ``BaseModelOutputWithPooling`` dataclass,
    matching how the SD1.5 pipeline consumes it: ``text_encoder(input_ids)[0]``.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids, return_dict=False)[0]


def shard_clip_text_encoder_specs(encoder) -> dict:
    """Megatron-style tensor-parallel shard specs for ``CLIPTextModel``.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, fc1): ("model", "batch")
    Row-parallel   (out_proj, fc2): ("batch", "model")

    ``encoder`` is the raw ``CLIPTextModel``. Layer norms and embeddings are
    left replicated. On a single device (mesh (1, 1)) these specs are a no-op.
    """
    specs = {}
    for layer in encoder.text_model.encoder.layers:
        attn = layer.self_attn
        specs[attn.q_proj.weight] = ("model", "batch")
        specs[attn.k_proj.weight] = ("model", "batch")
        specs[attn.v_proj.weight] = ("model", "batch")
        specs[attn.out_proj.weight] = ("batch", "model")

        specs[layer.mlp.fc1.weight] = ("model", "batch")
        specs[layer.mlp.fc2.weight] = ("batch", "model")

    return specs


class ModelVariant(StrEnum):
    """Available Stable Diffusion 1.5 model variants."""

    BASE = "Base"
    TEXT_ENCODER = "TextEncoder"


class ModelLoader(ForgeModel):
    """Stable Diffusion 1.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5",
        ),
        ModelVariant.TEXT_ENCODER: ModelConfig(
            pretrained_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with the requested variant.

        Args:
            variant: Optional ``ModelVariant``; falls back to ``DEFAULT_VARIANT``.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="Stable Diffusion 1.5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SD1.5 UNet.

        Args:
            dtype_override: Optional ``torch.dtype`` for the UNet weights;
                defaults to ``torch.bfloat16`` to match TT execution.

        Returns:
            torch.nn.Module: The ``UNet2DConditionModel`` instance for SD1.5,
            or a ``CLIPTextEncoderWrapper`` around the CLIP-L text encoder for
            the ``TEXT_ENCODER`` variant.
        """
        dtype = dtype_override or torch.bfloat16

        if self._variant == ModelVariant.TEXT_ENCODER:
            text_encoder = CLIPTextModel.from_pretrained(
                CLIP_REPO_ID, torch_dtype=dtype, **kwargs
            )
            return CLIPTextEncoderWrapper(text_encoder).eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", **kwargs
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", **kwargs
        )
        unet = UNet2DConditionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=dtype,
            **kwargs,
        )
        self.scheduler = LMSDiscreteScheduler.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="scheduler",
            **kwargs,
        )

        self.in_channels = unet.in_channels
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return a single-step UNet sample input batch for SD1.5.

        Args:
            dtype_override: Optional ``torch.dtype``; defaults to ``torch.bfloat16``.
            batch_size: Repetition factor for the prompt.

        Returns:
            dict: ``{"sample": …, "timestep": 0, "encoder_hidden_states": …}``
            for the UNet, or ``[input_ids]`` for the ``TEXT_ENCODER`` variant.
        """
        dtype = dtype_override or torch.bfloat16

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, CLIP_VOCAB_SIZE, (batch_size, CLIP_MAX_SEQ_LEN), dtype=torch.long
            )
            return [input_ids]

        prompt = ["A fantasy landscape with mountains and rivers"] * batch_size
        text_input = self.tokenizer(prompt, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        height, width = 512, 512
        latents = torch.randn((batch_size, self.in_channels, height // 8, width // 8))

        num_inference_steps = 1
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        latent_model_input = self.scheduler.scale_model_input(latents, 0)

        return {
            "sample": latent_model_input.to(dtype),
            "timestep": 0,
            "encoder_hidden_states": text_embeddings.to(dtype),
        }

    def get_mesh_config(self, num_devices: int):
        """Return ``(mesh_shape, mesh_names)`` for a ("batch", "model") 2D mesh.

        Supported device counts: 1, 2, 4, 8, 32.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor → partition_spec dict for the active component.

        Only ``TEXT_ENCODER`` provides shard specs; the model object is the
        ``CLIPTextEncoderWrapper`` returned by :meth:`load_model`.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_clip_text_encoder_specs(model.encoder)
        return None
