# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SRPO (tencent/SRPO) model loader implementation.

SRPO is a FLUX.1-dev fine-tune from Tencent Hunyuan that publishes only the
transformer weights (``diffusion_pytorch_model.safetensors``). Rather than
adding it as a variant of the existing ``flux`` loader, this introduces a
dedicated loader package so its preprocessing tweaks, license-gated weights,
and bringup state can evolve independently. This mirrors the layout used by
``stable_diffusion_3`` and ``bria_2_3``.

``load_model`` returns the FLUX transformer (with SRPO weights overlaid) as
an ``nn.Module``. ``load_inputs`` returns the positional tensors the FLUX
transformer consumes — the same shape contract as ``flux/pytorch/loader.py``.

Reference: https://huggingface.co/tencent/SRPO
"""

from typing import Optional

import torch
from transformers import CLIPTextModel, T5EncoderModel

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
from .src.model_utils import load_pipe, srpo_preprocessing
from .src.shard_specs import build_shard_spec, get_mesh_shape

# FLUX.1-dev's (and thus SRPO's) CLIP text encoder is CLIP-L, i.e. the standard
# ``openai/clip-vit-large-patch14`` ``CLIPTextModel``. We load it directly here
# rather than from the gated ``black-forest-labs/FLUX.1-dev`` ``text_encoder``
# subfolder so the component can be exercised without FLUX license access; the
# weights/architecture are identical (matches ``stable_diffusion_1_5``).
CLIP_REPO_ID = "openai/clip-vit-large-patch14"
# CLIP context length (tokenizer_max_length) and vocab size.
CLIP_MAX_SEQ_LEN = 77
CLIP_VOCAB_SIZE = 49408


class CLIPTextEncoderWrapper(torch.nn.Module):
    """Run ``CLIPTextModel`` as a stateless encoder returning a plain tensor.

    Pins ``return_dict=False`` so graph capture sees a pure tensor rather than a
    ``BaseModelOutputWithPooling`` dataclass, and returns the ``pooler_output``
    (index 1) — mirroring how SRPO's FLUX pipeline consumes the CLIP tower:
    ``pipe.text_encoder(input_ids, output_hidden_states=False).pooler_output``
    (see ``srpo_preprocessing``).
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids, return_dict=False)[1]


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


# FLUX.1-dev's (and thus SRPO's) second text encoder is the frozen T5-v1.1-XXL
# ``T5EncoderModel`` (google/t5-v1_1-xxl, ~4.7B). It is the same off-the-shelf
# encoder SD3 Medium ships as ``text_encoder_3``, so we load it from that repo's
# subfolder — ungated and identical weights — to avoid the gated FLUX.1-dev
# ``text_encoder_2`` (mirrors the CLIP-provenance choice above and reuses the
# ``stable_diffusion_3`` T5 component).
T5_REPO_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
T5_SUBFOLDER = "text_encoder_3"
# T5 sequence length SRPO feeds the encoder (srpo_preprocessing max_sequence_length)
# and the T5 v1.1 SentencePiece vocab size.
T5_MAX_SEQ_LEN = 256
T5_VOCAB_SIZE = 32128


class T5TextEncoderWrapper(torch.nn.Module):
    """Run ``T5EncoderModel`` as a stateless encoder returning a plain tensor.

    Pins ``return_dict=False`` so graph capture sees a pure tensor
    (last_hidden_state) rather than a ``BaseModelOutput`` dataclass — mirroring
    how SRPO's FLUX pipeline consumes the T5 tower:
    ``pipe.text_encoder_2(input_ids, output_hidden_states=False)[0]``
    (see ``srpo_preprocessing``).
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids, return_dict=False)[0]


def shard_t5_text_encoder_specs(encoder) -> dict:
    """Megatron-style tensor-parallel shard specs for ``T5EncoderModel``.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, wi_0, wi_1): ("model", "batch")
    Row-parallel   (o, wo):                ("batch", "model")

    ``encoder`` is the raw ``T5EncoderModel``. Layer norms and the relative
    attention bias are left replicated. On a single device (mesh (1, 1)) these
    specs are a no-op.
    """
    specs = {encoder.shared.weight: (None, "batch")}
    for block in encoder.encoder.block:
        attn = block.layer[0].SelfAttention
        specs[attn.q.weight] = ("model", "batch")
        specs[attn.k.weight] = ("model", "batch")
        specs[attn.v.weight] = ("model", "batch")
        specs[attn.o.weight] = ("batch", "model")

        ff = block.layer[1].DenseReluDense
        specs[ff.wi_0.weight] = ("model", "batch")
        specs[ff.wi_1.weight] = ("model", "batch")
        specs[ff.wo.weight] = ("batch", "model")

    return specs


class ModelVariant(StrEnum):
    """Available SRPO model variants."""

    BASE = "Base"
    TEXT_ENCODER = "TextEncoder"
    T5_TEXT_ENCODER = "T5TextEncoder"


class ModelLoader(ForgeModel):
    """SRPO model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="tencent/SRPO",
        ),
        ModelVariant.TEXT_ENCODER: ModelConfig(
            pretrained_model_name=CLIP_REPO_ID,
        ),
        ModelVariant.T5_TEXT_ENCODER: ModelConfig(
            pretrained_model_name=T5_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Prompt taken from the SRPO Hugging Face model card.
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given SRPO variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.pipe = None
        # SRPO inherits FLUX.1-dev's guidance scale (3.5 per the model card).
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in (ModelVariant.TEXT_ENCODER, ModelVariant.T5_TEXT_ENCODER)
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="SRPO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load (and cache) the SRPO pipeline (FLUX.1-dev base + SRPO weights)."""
        self.pipe = load_pipe(
            self._variant_config.pretrained_model_name,
            dtype_override=dtype_override,
        )
        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the SRPO transformer (FLUX.1-dev architecture, SRPO weights).

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the pipeline to.

        Returns:
            torch.nn.Module: The FLUX transformer with SRPO weights overlaid,
            or a ``CLIPTextEncoderWrapper`` around the CLIP-L text encoder for
            the ``TEXT_ENCODER`` variant.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            dtype = dtype_override if dtype_override is not None else torch.bfloat16
            encoder = CLIPTextModel.from_pretrained(CLIP_REPO_ID, torch_dtype=dtype)
            return CLIPTextEncoderWrapper(encoder).eval()

        if self._variant == ModelVariant.T5_TEXT_ENCODER:
            dtype = dtype_override if dtype_override is not None else torch.bfloat16
            encoder = T5EncoderModel.from_pretrained(
                T5_REPO_ID, subfolder=T5_SUBFOLDER, torch_dtype=dtype, device_map="cpu"
            )
            return T5TextEncoderWrapper(encoder).eval()

        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)
        elif dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return positional inputs for the FLUX transformer (SRPO weights).

        Args:
            dtype_override: Optional ``torch.dtype`` for the returned tensors.
            batch_size: Batch size for the synthetic input. Defaults to 1.

        Returns:
            dict: Input tensors that can be fed directly to the transformer
            (matches the keyword-argument signature of FLUX's
            ``FluxTransformer2DModel.forward``). For ``TEXT_ENCODER`` returns
            ``[input_ids]`` for the CLIP-L encoder.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, CLIP_VOCAB_SIZE, (batch_size, CLIP_MAX_SEQ_LEN), dtype=torch.long
            )
            return [input_ids]

        if self._variant == ModelVariant.T5_TEXT_ENCODER:
            input_ids = torch.randint(
                0, T5_VOCAB_SIZE, (batch_size, T5_MAX_SEQ_LEN), dtype=torch.long
            )
            return [input_ids]

        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        (
            hidden_states,
            timestep,
            guidance,
            pooled_projections,
            encoder_hidden_states,
            txt_ids,
            img_ids,
        ) = srpo_preprocessing(
            self.pipe,
            self.prompt,
            dtype=dtype,
            batch_size=batch_size,
            guidance_scale=self.guidance_scale,
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_projections,
            "encoder_hidden_states": encoder_hidden_states,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "joint_attention_kwargs": {},
        }

    def get_mesh_config(self, num_devices: int):
        """Return ``(mesh_shape, mesh_names)`` for tensor-parallel execution.

        SRPO is a ~12B FLUX DiT that runs out of DRAM on a single chip; it is
        brought up across multiple chips with Megatron-1D tensor parallelism
        over a ``(None, "model")`` mesh. See ``src/shard_specs.py``.

        Args:
            num_devices: Total chip count (``xr.global_runtime_device_count()``).

        Returns:
            tuple: ``(mesh_shape, mesh_names)`` consumed by the auto-runner.
        """
        return get_mesh_shape(num_devices)

    def load_shard_spec(self, model):
        """Return the tensor -> partition-spec mapping for the SRPO transformer.

        Args:
            model: the ``FluxTransformer2DModel`` returned by ``load_model``.

        Returns:
            dict: ``{torch.nn.Parameter: partition_spec}``. Parameters absent
            from the mapping are replicated across the mesh.
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_clip_text_encoder_specs(model.encoder)
        if self._variant == ModelVariant.T5_TEXT_ENCODER:
            return shard_t5_text_encoder_specs(model.encoder)
        return build_shard_spec(model)
