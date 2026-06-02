# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 model loader implementation for text-to-image generation.

FLUX.2 uses a new transformer architecture (``Flux2Transformer2DModel``) that
differs from FLUX.1: a Mistral3 multimodal text encoder produces a high
dimensional joint-attention conditioning (``joint_attention_dim=15360``), there
is no separate pooled CLIP projection, and rotary position embeddings use 4
coordinate axes (T, H, W, L).

This loader exercises the transformer (the heavy, compute-dominant component of
the pipeline) in isolation. The text-encoder conditioning is synthesized
directly at the transformer's expected ``joint_attention_dim`` so the ~24B
Mistral3 text encoder does not have to be downloaded or run for bringup; the
transformer forward only depends on the shapes/dtypes of its inputs, and the
golden-vs-device PCC comparison uses the same synthesized inputs on both sides.

The FLUX.2-dev transformer has ~32B parameters (~64 GB in bfloat16), which does
not fit on a single chip. The loader therefore exposes a tensor-parallel mesh
config and weight shard specs (Megatron-style column/row partitioning) so the
transformer can be sharded across multiple chips via the test runner's
``tensor_parallel`` mode.
"""
import torch
from typing import Optional

from diffusers import Flux2Transformer2DModel

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available FLUX.2 model variants."""

    DEV = "Dev"


# (batch, model) mesh shapes by device count. FLUX.2-dev is sharded along the
# "model" axis (tensor parallel); the "batch" axis is kept at 1.
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}
MESH_NAMES = ("batch", "model")


class Flux2TransformerWrapper(torch.nn.Module):
    """Flatten the FLUX.2 transformer forward to keyword tensors and return a
    single output tensor (the predicted noise) instead of an output dataclass.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        guidance,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]


def shard_flux2_transformer_specs(transformer) -> dict:
    """Tensor → partition_spec map for ``Flux2Transformer2DModel``.

    Mesh axes: ("batch", "model"). Tensor parallel shards along "model".

    Every 2D weight ``(out, in)`` is sharded along its *input* dimension
    (row-parallel, ``("batch", "model")``). This splits all weight matrices
    ``model``-ways for memory, while keeping every activation full/replicated at
    op boundaries: the matmul produces partial sums that are all-reduced back to
    a full tensor before the next op.

    Row-parallel everywhere (rather than the usual column/row Megatron mix) is
    deliberate for FLUX.2: several projections fuse gate+value (SwiGLU
    ``ff.linear_in``), fuse QKV+MLP (``to_qkv_mlp_proj``), or feed ``chunk()`` /
    ``split()`` for modulation and adaLN. Column-sharding their *output* would
    place the two halves of a fused/gated tensor on different devices, breaking
    the elementwise pairing after the chunk. Sharding the input dimension keeps
    those outputs whole, so the fused chunks stay correctly aligned.

    1D RMSNorm weights are tiny and left replicated (omitted from the spec).
    All FLUX.2 transformer linears are bias-free, so only weights are sharded.
    """
    specs = {}
    ROW = ("batch", "model")  # shard the (out, in) weight along its input dim

    # Input projections.
    specs[transformer.x_embedder.weight] = ROW  # (inner, in_ch)
    specs[transformer.context_embedder.weight] = ROW  # (inner, joint)

    # Combined timestep + guidance embedding (each is an MLP: linear_1 -> linear_2).
    tg = transformer.time_guidance_embed
    specs[tg.timestep_embedder.linear_1.weight] = ROW
    specs[tg.timestep_embedder.linear_2.weight] = ROW
    if tg.guidance_embedder is not None:
        specs[tg.guidance_embedder.linear_1.weight] = ROW
        specs[tg.guidance_embedder.linear_2.weight] = ROW

    # Shared modulation projections (outputs feed Flux2Modulation.split -> chunk).
    specs[transformer.double_stream_modulation_img.linear.weight] = ROW
    specs[transformer.double_stream_modulation_txt.linear.weight] = ROW
    specs[transformer.single_stream_modulation.linear.weight] = ROW

    # Double-stream blocks: joint image+text attention + two SwiGLU feedforwards.
    for block in transformer.transformer_blocks:
        attn = block.attn
        specs[attn.to_q.weight] = ROW
        specs[attn.to_k.weight] = ROW
        specs[attn.to_v.weight] = ROW
        specs[attn.to_out[0].weight] = ROW
        specs[attn.add_q_proj.weight] = ROW
        specs[attn.add_k_proj.weight] = ROW
        specs[attn.add_v_proj.weight] = ROW
        specs[attn.to_add_out.weight] = ROW

        specs[block.ff.linear_in.weight] = ROW  # fused gate+value (SwiGLU)
        specs[block.ff.linear_out.weight] = ROW
        specs[block.ff_context.linear_in.weight] = ROW  # fused gate+value (SwiGLU)
        specs[block.ff_context.linear_out.weight] = ROW

    # Single-stream blocks: fused QKV+MLP-in and fused attn-out+MLP-out.
    for block in transformer.single_transformer_blocks:
        attn = block.attn
        specs[attn.to_qkv_mlp_proj.weight] = ROW  # fused QKV + MLP-in (SwiGLU)
        specs[attn.to_out.weight] = ROW  # fused attn-out + MLP-out

    # Output layers.
    specs[transformer.norm_out.linear.weight] = ROW  # feeds adaLN chunk
    specs[transformer.proj_out.weight] = ROW

    return specs


class ModelLoader(ForgeModel):
    """FLUX.2 transformer loader for text-to-image generation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.DEV: ModelConfig(
            pretrained_model_name="black-forest-labs/FLUX.2-dev",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEV

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.transformer = None
        # FLUX.2-dev is a guidance-distilled model.
        self.guidance_scale = 4.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLUX.2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,  # text-to-image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the FLUX.2 transformer model for this instance's variant.

        Only the transformer subfolder is fetched/loaded; the Mistral3 text
        encoder and VAE are not required to exercise the transformer. The
        transformer is wrapped so its forward takes plain keyword tensors and
        returns a single output tensor.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses bfloat16.

        Returns:
            torch.nn.Module: The wrapped FLUX.2 transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        transformer = Flux2Transformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
            torch_dtype=dtype,
            use_safetensors=True,
        )
        transformer.eval()
        self.transformer = transformer

        return Flux2TransformerWrapper(transformer)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FLUX.2 transformer.

        Inputs are constructed to match the shapes/dtypes the transformer
        expects, mirroring how ``Flux2Pipeline`` prepares latents, conditioning
        embeddings and 4D rotary position ids, but at a small spatial/sequence
        size suitable for bringup.

        Args:
            dtype_override: Optional torch.dtype to override the default dtype.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors that can be fed to the wrapped transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Transformer config-derived dimensions for FLUX.2-dev.
        in_channels = 128  # transformer.config.in_channels
        joint_attention_dim = 15360  # transformer.config.joint_attention_dim

        # Small bringup sizes.
        # Image latent grid (already packed: these are height//2, width//2).
        latent_h = 16
        latent_w = 16
        image_seq_len = latent_h * latent_w
        # Text conditioning sequence length (real pipeline uses 512).
        text_seq_len = 128

        # Image latents: (B, image_seq_len, in_channels)
        hidden_states = torch.randn(
            batch_size, image_seq_len, in_channels, dtype=dtype
        )

        # Text conditioning embeddings: (B, text_seq_len, joint_attention_dim)
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        # 4D rotary position ids (T, H, W, L) for image tokens, matching
        # Flux2Pipeline._prepare_latent_ids: T=0, H in [0,latent_h), W in
        # [0,latent_w), L=0.
        img_ids = torch.cartesian_prod(
            torch.arange(1),
            torch.arange(latent_h),
            torch.arange(latent_w),
            torch.arange(1),
        ).to(dtype=torch.float32)
        img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        # 4D rotary position ids for text tokens, matching
        # Flux2Pipeline._prepare_text_ids: T=0, H=0, W=0, L in [0,text_seq_len).
        txt_ids = torch.cartesian_prod(
            torch.arange(1),
            torch.arange(1),
            torch.arange(1),
            torch.arange(text_seq_len),
        ).to(dtype=torch.float32)
        txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        # Denoising timestep (normalized to [0, 1]) and distilled guidance scale.
        timestep = torch.full((batch_size,), 0.5, dtype=dtype)
        guidance = torch.full((batch_size,), self.guidance_scale, dtype=torch.float32)

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,
        }

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        FLUX.2-dev is sharded along the "model" axis. Supported device counts:
        1, 2, 4, 8.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor → partition_spec dict for the FLUX.2 transformer.

        Expects the wrapped model returned by load_model(); specs are built from
        ``model.transformer``.
        """
        transformer = model.transformer if hasattr(model, "transformer") else model
        return shard_flux2_transformer_specs(transformer)
