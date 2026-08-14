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
        """Return positional forward-pass inputs for the Infinity transformer.

        Delegates input construction to
        :func:`.src.model_utils.build_forward_inputs`. Targets the
        training-style ``forward`` path (``cfg_infer=False``) -- a single
        traceable pass that returns logits, not a sampling loop.

        Args:
            dtype_override: Optional ``torch.dtype`` for the tensor inputs.
            batch_size: Replication factor for the prompt.
            prompt: Optional override prompt; defaults to a fixed string.

        Returns:
            list: Positional ``forward`` arguments in signature order
                ``[label_B_or_BLT, x_BLC_wo_prefix, scale_schedule]``, where
                ``label_B_or_BLT`` is ``(kv_compact, lens, cu_seqlens_k,
                max_seqlen_k)``. A positional sequence (not a dict) is required
                because the test infra invokes the model as ``model(*inputs)``;
                unpacking a dict would pass its string keys instead of tensors.
        """
        if self.text_encoder is None:
            raise RuntimeError("load_model() must be called before load_inputs().")
        forward_inputs = build_forward_inputs(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            pn=self._PN,
            batch_size=batch_size,
            prompt=prompt,
            dtype_override=dtype_override,
        )
        return [
            forward_inputs["label_B_or_BLT"],
            forward_inputs["x_BLC_wo_prefix"],
            forward_inputs["scale_schedule"],
        ]

    def get_mesh_config(self, num_devices: int):
        """Mesh config for tensor-parallel sharding of the Infinity transformer.

        Uses the mochi-style 2D mesh ``(1, 8)`` with axis names
        ``(None, "model")``: the 8-wide ``model`` axis (index 1) carries the
        tensor parallelism and the size-1 axis is named ``None`` so no partition
        spec ever references it. ``load_inputs`` uses ``batch_size=1`` so there
        is no data-parallel axis. An 8-wide ``model`` axis splits the 16
        attention heads 2-per-device, shrinking the O(L^2) self-attention score
        tensor from ``[1, 16, L, L]`` to ``[1, 2, L, L]`` -- the buffer that
        OOM'd when attention was replicated.

        Args:
            num_devices: Total devices visible to the runtime.

        Returns:
            tuple: ``(mesh_shape, axis_names)``.
        """
        if num_devices == 8:
            mesh_shape = (1, 8)
        else:
            raise ValueError(
                f"Infinity sharding currently supports 8 devices, got {num_devices}."
            )
        return mesh_shape, (None, "model")

    def load_shard_spec(self, model):
        """Megatron tensor-parallel shard spec for the Infinity transformer.

        Head-parallel attention: each of the (unfused) q/k/v projections is
        column-parallel (split output heads on ``model``), and the output
        ``proj`` is row-parallel (split the contraction dim) -- one all-reduce
        per attention/FFN pair (Megatron-LM, arXiv:1909.08053). The projections
        are unfused in ``model.py`` (``mat_qkv`` -> ``mat_q/mat_k/mat_v``,
        ``mat_kv`` -> ``mat_k/mat_v``) precisely so a single partition spec can
        place matching per-head q/k/v on the same device -- sharding the fused
        qkv-major weight directly is numerically wrong (PCC ~ -0.18). Mesh is
        the mochi ``(1, 8)`` / ``(None, "model")``.

        Per-head scale (``scale_mul_1H11``), the concatenated bias buffers,
        norms, lvl/positional embeddings and the tiny head are left replicated;
        the partitioner slices them to match the sharded heads.

        Args:
            model: The Infinity transformer instance.

        Returns:
            Dict[torch.Tensor, tuple]: parameter -> partition spec.
        """
        specs = {}
        for block in model.unregistered_blocks:
            # --- self-attention: column-parallel q/k/v, row-parallel proj ---
            sa = block.sa
            specs[sa.mat_q.weight] = ("model", None)
            specs[sa.mat_k.weight] = ("model", None)
            specs[sa.mat_v.weight] = ("model", None)
            specs[sa.proj.weight] = (None, "model")  # row: all-reduce
            if sa.proj.bias is not None:
                specs[sa.proj.bias] = (None,)

            # --- cross-attention: column-parallel q/k/v, row-parallel proj ---
            ca = block.ca
            specs[ca.mat_q.weight] = ("model", None)
            if ca.mat_q.bias is not None:
                specs[ca.mat_q.bias] = ("model",)
            specs[ca.mat_k.weight] = ("model", None)
            specs[ca.mat_v.weight] = ("model", None)
            specs[ca.proj.weight] = (None, "model")  # row: all-reduce
            if ca.proj.bias is not None:
                specs[ca.proj.bias] = (None,)

            # --- FFN (column fc1 -> row fc2) ---
            specs[block.ffn.fc1.weight] = ("model", None)
            if block.ffn.fc1.bias is not None:
                specs[block.ffn.fc1.bias] = ("model",)
            specs[block.ffn.fc2.weight] = (None, "model")
            if block.ffn.fc2.bias is not None:
                specs[block.ffn.fc2.bias] = (None,)
        return specs
