# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PI0.5 (Pi05) model loader implementation for vision-language-action prediction.

PI0.5 is a Vision-Language-Action (VLA) policy from Physical Intelligence,
distributed through the ``lerobot`` library. It pairs a PaliGemma backbone
(Gemma-2B language model + SigLIP vision tower) with a Gemma-300M "action
expert" and predicts continuous action chunks via flow matching.

The full inference path (``sample_actions``) runs an iterative flow-matching
denoising loop with data-dependent control flow, which is not a single static
graph. For bringup we expose the model's core *single forward pass*
(``PI05Pytorch.forward``-equivalent): vision tower + language embedding +
joint PaliGemma/expert transformer + action projection, producing the
predicted velocity field ``v_t`` for one flow-matching step. This is the
compute-dominant component (vision tower + both Gemma stacks) and is a single
deterministic graph suitable for compile/PCC validation.
"""

from typing import Optional

import torch
import torch.nn as nn

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available PI0.5 model variants."""

    BASE = "base"


class PI05ForwardWrapper(nn.Module):
    """Single-forward-pass wrapper around the lerobot ``PI05Pytorch`` core model.

    Exposes a flat-tensor ``forward`` (one tensor per camera, plus language /
    state / flow-matching tensors) and returns the predicted velocity field
    ``v_t`` of shape ``[batch, chunk_size, max_action_dim]`` — the model's
    actual flow-matching output. The body mirrors ``PI05Pytorch.forward`` but
    returns ``v_t`` instead of the training MSE loss, so the runner can
    PCC-compare a deterministic, meaningful tensor.
    """

    def __init__(self, core: nn.Module, num_cameras: int, chunk_size: int):
        super().__init__()
        self.core = core  # lerobot PI05Pytorch
        self.num_cameras = num_cameras
        self.chunk_size = chunk_size

    def forward(
        self,
        image_0,
        image_1,
        image_2,
        img_mask_0,
        img_mask_1,
        img_mask_2,
        tokens,
        masks,
        actions,
        noise,
        time,
    ):
        # Imported here so the module import does not hard-require lerobot until
        # the model is actually instantiated.
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        m = self.core
        images = [image_0, image_1, image_2][: self.num_cameras]
        img_masks = [img_mask_0, img_mask_1, img_mask_2][: self.num_cameras]

        # Construct the noisy action sample x_t for this flow-matching timestep.
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = m.embed_suffix(
            x_t, time
        )

        # Match the original forward's dtype alignment (weights may be bf16).
        if (
            m.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # cumsum over a bool mask: cast to int32 first. PyTorch promotes
        # bool->int64 implicitly on CPU, but the TT path rejects subtraction on
        # a bool tensor (`cumsum(bool) - 1`), so make the integer dtype explicit.
        position_ids = torch.cumsum(pad_masks.to(torch.int32), dim=1) - 1
        att_2d_masks_4d = m._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = m.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = m.action_out_proj(suffix_out)
        return v_t


class ModelLoader(ForgeModel):
    """PI0.5 (Pi05) VLA model loader for action-prediction tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="lerobot/pi05_base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Sample task instruction used to build the language prefix.
    sample_instruction = "pick up the cube and place it in the box"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.policy = None
        self.cfg = None
        # Number of camera views from the checkpoint config (base / left / right wrist).
        self.num_cameras = 3

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="pi05",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _precision_from_dtype(dtype_override):
        """Map a torch dtype override to the lerobot PI05Config precision string."""
        if dtype_override == torch.bfloat16:
            return "bfloat16"
        # Default / float32 — the checkpoint's native precision.
        return "float32"

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the PI0.5 core model wrapped for a single forward pass.

        Args:
            dtype_override: Optional torch.dtype. ``torch.bfloat16`` loads the
                model in its mixed-precision bf16 mode (vision tower / projector
                / norms are kept in float32 by the model itself); anything else
                (or None) loads the native float32 weights.

        Returns:
            torch.nn.Module: PI05ForwardWrapper around the core PI05Pytorch model.
        """
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05 import PI05Policy

        pretrained_model_name = self._variant_config.pretrained_model_name

        cfg = PreTrainedConfig.from_pretrained(pretrained_model_name)
        # The checkpoint config requests device "mps"; force CPU loading so the
        # weights land on host before the runner moves them to the TT device.
        cfg.device = "cpu"
        cfg.dtype = self._precision_from_dtype(dtype_override)

        policy = PI05Policy.from_pretrained(pretrained_model_name, config=cfg)
        policy = policy.eval()

        self.policy = policy
        self.cfg = cfg

        model = PI05ForwardWrapper(
            core=policy.model,
            num_cameras=self.num_cameras,
            chunk_size=cfg.chunk_size,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Build a deterministic sample observation batch for one forward pass.

        Returns a dict of flat tensors matching ``PI05ForwardWrapper.forward``:
        three camera images, their presence masks, language tokens + pad mask,
        and the flow-matching ``actions`` / ``noise`` / ``time`` tensors.

        Inputs are synthetic but deterministic (fixed seed). Real robot
        observations require a LeRobotDataset and the policy pre-processor; for
        bringup we validate the compute graph with representative shapes.

        Args:
            dtype_override: Optional torch.dtype. When ``torch.bfloat16``, the
                float-valued image tensors are produced in bf16 so they match
                the model's bf16 image-embedding path; otherwise float32.
            batch_size: Batch size (default 1).
        """
        if self.cfg is not None:
            chunk_size = self.cfg.chunk_size
            action_dim = self.cfg.max_action_dim
            tok_len = self.cfg.tokenizer_max_length
        else:
            # Defaults matching lerobot/pi05_base config.json.
            chunk_size, action_dim, tok_len = 50, 32, 200

        # Vision tower internally casts images to float32, so bf16 only matters
        # for keeping the image-embedding dtype consistent with bf16 weights.
        img_dtype = torch.bfloat16 if dtype_override == torch.bfloat16 else torch.float32

        gen = torch.Generator().manual_seed(0)
        B = batch_size

        images = [
            torch.rand(B, 3, 224, 224, generator=gen, dtype=torch.float32).to(img_dtype)
            for _ in range(self.num_cameras)
        ]
        img_masks = [torch.ones(B, dtype=torch.bool) for _ in range(self.num_cameras)]

        # Language prefix: valid PaliGemma vocab ids (vocab size 257152).
        tokens = torch.randint(0, 257152, (B, tok_len), generator=gen, dtype=torch.long)
        masks = torch.ones(B, tok_len, dtype=torch.bool)

        actions = torch.randn(B, chunk_size, action_dim, generator=gen, dtype=torch.float32)
        noise = torch.randn(B, chunk_size, action_dim, generator=gen, dtype=torch.float32)
        # Flow-matching timestep in (0, 1).
        time = torch.rand(B, generator=gen, dtype=torch.float32) * 0.998 + 0.001

        return {
            "image_0": images[0],
            "image_1": images[1],
            "image_2": images[2],
            "img_mask_0": img_masks[0],
            "img_mask_1": img_masks[1],
            "img_mask_2": img_masks[2],
            "tokens": tokens,
            "masks": masks,
            "actions": actions,
            "noise": noise,
            "time": time,
        }
