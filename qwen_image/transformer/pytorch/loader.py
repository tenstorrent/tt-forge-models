# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image transformer (MMDiT denoiser) loader for text-to-image generation.

Qwen-Image is a multi-stage diffusion pipeline (Qwen2.5-VL text encoder, an
``AutoencoderKLQwenImage`` VAE, and the ``QwenImageTransformer2DModel`` MMDiT
denoiser driven by a FlowMatchEulerDiscreteScheduler). This loader brings up the
**denoiser** — the per-step compute and the heavy sharding target. The VAE and
text-encoder components have their own sibling loaders.
"""
import torch
from typing import Optional

from diffusers import QwenImageTransformer2DModel

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
    """Available Qwen-Image transformer variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Qwen-Image MMDiT denoiser loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Qwen/Qwen-Image",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Native generation resolution (pipeline default — 1024x1024).
    HEIGHT = 1024
    WIDTH = 1024
    # Representative text-prompt sequence length (Qwen2.5-VL prompt embeds).
    TEXT_SEQ_LEN = 256

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="qwen_image_transformer",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Qwen-Image transformer (denoiser).

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype (the weights are distributed in bfloat16).

        Returns:
            torch.nn.Module: The QwenImageTransformer2DModel instance.
        """
        model_kwargs = {"subfolder": "transformer", "use_safetensors": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = QwenImageTransformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model = model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the Qwen-Image denoiser at native resolution.

        The denoiser consumes packed image latents plus the Qwen2.5-VL prompt
        embeddings. The prompt embeddings are synthesized here (a representative
        random tensor of the correct shape) so the denoiser can be validated
        without instantiating the 7B text encoder; the composite pipeline feeds
        the real embeddings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors for QwenImageTransformer2DModel.forward.
        """
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        config = self.model.config
        # in_channels (64) is the packed latent channel count (16 latent * 2 * 2).
        in_channels = config.in_channels
        joint_attention_dim = config.joint_attention_dim  # 3584

        # VAE scale factor for AutoencoderKLQwenImage is 8 (2 ** 3 temporal
        # downsample levels); the pipeline packs 2x2 patches on top of that.
        vae_scale_factor = 8
        latent_h = self.HEIGHT // vae_scale_factor // 2  # 64
        latent_w = self.WIDTH // vae_scale_factor // 2  # 64
        img_seq_len = latent_h * latent_w  # 4096

        # Packed image latents: [B, img_seq_len, in_channels]
        hidden_states = torch.randn(
            batch_size, img_seq_len, in_channels, dtype=dtype
        )

        # Qwen2.5-VL prompt embeddings: [B, text_seq_len, joint_attention_dim]
        encoder_hidden_states = torch.randn(
            batch_size, self.TEXT_SEQ_LEN, joint_attention_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(
            batch_size, self.TEXT_SEQ_LEN, dtype=torch.int64
        )

        # FlowMatch timestep in [0, 1] (pipeline feeds t / 1000).
        timestep = torch.full((batch_size,), 0.9, dtype=dtype)

        # img_shapes: list (per batch) of (frames, latent_h, latent_w) tuples.
        img_shapes = [[(1, latent_h, latent_w)]] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "guidance": None,
            "return_dict": False,
        }

    def get_mesh_config(self, num_devices: int):
        """Mesh for tensor-parallel sharding of the MMDiT denoiser.

        The denoiser is ~20B params (~40GB in bf16) and does not fit a single
        32GB Blackhole chip, so a tensor-parallel layout across the device's
        chips is the baseline. Uses a 1xN mesh (batch x model).

        Args:
            num_devices: Number of devices to distribute the model across.

        Returns:
            (mesh_shape, mesh_names): e.g. ((1, num_devices), ("batch", "model")).
        """
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:
            mesh_shape = (1, num_devices)

        assert self.model is not None, "Call load_model() before get_mesh_config()"
        assert (
            self.model.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron column->row shard spec for the joint-attention MMDiT blocks.

        Each ``QwenImageTransformerBlock`` runs joint attention over the image
        and text streams plus a per-stream FeedForward. Column-shard the q/k/v
        projections (both image and text-added) and the FFN up-projection on the
        ``model`` axis; row-shard the attention output and FFN down-projection.

        Args:
            model: The model instance (on device).

        Returns:
            Dict[Tensor, Tuple[str, str]]: weight -> (dim0 axis, dim1 axis).
        """
        shard_specs = {}
        for block in model.transformer_blocks:
            attn = block.attn
            # Joint-attention QKV projections (image + text streams): column-parallel.
            for proj in (
                attn.to_q,
                attn.to_k,
                attn.to_v,
                attn.add_q_proj,
                attn.add_k_proj,
                attn.add_v_proj,
            ):
                shard_specs[proj.weight] = ("model", "batch")
            # Attention output projections: row-parallel.
            shard_specs[attn.to_out[0].weight] = ("batch", "model")
            shard_specs[attn.to_add_out.weight] = ("batch", "model")

            # Per-stream FeedForward: up-proj column, down-proj row.
            for mlp in (block.img_mlp, block.txt_mlp):
                shard_specs[mlp.net[0].proj.weight] = ("model", "batch")
                shard_specs[mlp.net[2].weight] = ("batch", "model")

            # AdaLN modulation (Sequential(SiLU, Linear) whose wide 18432 output
            # is .chunk()ed into shift/scale/gate applied elementwise on the full
            # hidden dim): ROW-shard the contraction dim so each chip computes a
            # partial output that is all-reduced into a replicated full vector.
            # Column-sharding it instead forces an all-gather that tt-mlir lowers
            # to a ttnn.concat whose CB page size exceeds Blackhole per-core L1 at
            # opt 0; replicating these ~6.8B modulation params OOMs a 32GB chip.
            shard_specs[block.img_mod[1].weight] = ("batch", "model")
            shard_specs[block.txt_mod[1].weight] = ("batch", "model")

        # Final AdaLN modulation before proj_out: row-shard for the same reason.
        # img_in / txt_in / proj_out stay replicated (not in the spec) so block
        # activations enter and leave the column/row-parallel blocks replicated.
        shard_specs[model.norm_out.linear.weight] = ("batch", "model")
        return shard_specs
