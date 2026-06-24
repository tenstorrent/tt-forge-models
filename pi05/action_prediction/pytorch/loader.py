# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PI0.5 (Pi05) model loader implementation for vision-language-action prediction.

Pi05 is a flow-matching Vision-Language-Action (VLA) policy from Physical
Intelligence, ported to LeRobot. It pairs a PaliGemma backbone (SigLIP vision
tower + Gemma-2B language model) with a Gemma-300M "action expert" and predicts
continuous action chunks via flow matching.

The underlying ``PI05Pytorch`` core takes its inputs as Python lists
(``images``, ``img_masks``) plus several positional tensors, which the runner's
``model(**inputs)`` calling convention cannot express directly. We therefore
wrap the core in a thin module that exposes a flat keyword-argument forward and
reconstructs the lists internally. A single forward pass runs the full
PaliGemma + action-expert stack and returns the flow-matching velocity loss
tensor of shape ``[batch, chunk_size, max_action_dim]``.
"""

from typing import Optional

import torch

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Pi05 model variants."""

    BASE = "base"


class PI05Wrapper(torch.nn.Module):
    """Flat keyword-argument wrapper around the ``PI05Pytorch`` core.

    Two responsibilities:

    1. The core forward takes ``images``/``img_masks`` as Python lists (one
       entry per camera). This wrapper accepts the three camera tensors and
       masks as individual keyword arguments so the model can be driven by
       ``model(**inputs)``.
    2. The core ``PI05Pytorch.forward`` is a *training* forward: it hardcodes a
       ``suffix_out.to(torch.float32)`` cast before the final ``action_out_proj``
       and returns an ``F.mse_loss`` of mixed dtypes. That breaks under uniform
       bf16 device execution. We instead reimplement the inference forward —
       one flow-matching denoiser evaluation — returning the predicted velocity
       ``v_t`` of shape ``[batch, chunk_size, max_action_dim]`` in the model's
       native dtype, with no float32 hardcode.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        image_0,
        image_1,
        image_2,
        img_mask_0,
        img_mask_1,
        img_mask_2,
        input_ids,
        attention_mask,
        actions,
        noise,
        time,
    ):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        m = self.model
        images = [image_0, image_1, image_2]
        img_masks = [img_mask_0, img_mask_1, img_mask_2]

        # Construct the noisy action sample x_t at the given flow-matching time.
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # Embed multimodal prefix (images + language) and action/time suffix.
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, input_ids, attention_mask
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = m.embed_suffix(
            x_t, time
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # Cast the bool pad mask to int before cumsum: the TT backend keeps
        # cumsum-of-bool as bool, and bool subtraction (`- 1`) is unsupported.
        position_ids = torch.cumsum(pad_masks.to(torch.int32), dim=1) - 1
        att_2d_masks_4d = m._prepare_attention_masks_4d(att_2d_masks)

        # Joint PaliGemma + action-expert forward.
        (_, suffix_out), _ = m.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Project the action-token outputs to the predicted velocity field.
        suffix_out = suffix_out[:, -m.config.chunk_size :]
        v_t = m.action_out_proj(suffix_out)
        return v_t


class ModelLoader(ForgeModel):
    """Pi05 VLA model loader for action-prediction tasks."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="lerobot/pi05_base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.policy = None
        self.config = None

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

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Pi05 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                            dtype. If not provided, the model uses float32.

        Returns:
            torch.nn.Module: A PI05Wrapper around the Pi05 flow-matching core.
        """
        from lerobot.policies.pi05 import PI05Policy

        pretrained_model_name = self._variant_config.pretrained_model_name

        policy = PI05Policy.from_pretrained(pretrained_model_name)
        policy.eval()
        self.policy = policy
        self.config = policy.config

        model = PI05Wrapper(policy.model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the Pi05 model.

        Builds a deterministic single-step observation batch: three RGB camera
        views (base, left wrist, right wrist), a language-instruction token
        sequence, and the flow-matching action/noise/time tensors.

        Args:
            dtype_override: Optional torch.dtype applied to the floating-point
                            inputs (images, actions, noise, time).
            batch_size: Batch size to use (default 1).

        Returns:
            dict: Keyword inputs matching ``PI05Wrapper.forward``.
        """
        if batch_size is None:
            batch_size = 1

        if self.config is None:
            # Ensure config is available (image_resolution, chunk_size, etc.).
            self.load_model(dtype_override=dtype_override)

        cfg = self.config
        h, w = cfg.image_resolution
        chunk = cfg.chunk_size
        act_dim = cfg.max_action_dim
        seq_len = cfg.tokenizer_max_length

        gen = torch.Generator().manual_seed(0)

        def rand(*shape):
            return torch.rand(*shape, generator=gen)

        # Three camera views, normalized to SigLIP's expected [-1, 1] range.
        images = [rand(batch_size, 3, h, w) * 2.0 - 1.0 for _ in range(3)]
        img_masks = [
            torch.ones(batch_size, dtype=torch.bool) for _ in range(3)
        ]

        # Language instruction tokens + attention mask.
        input_ids = torch.randint(
            0, 1000, (batch_size, seq_len), generator=gen, dtype=torch.long
        )
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Flow-matching action / noise / time tensors.
        actions = torch.zeros(batch_size, chunk, act_dim)
        noise = torch.randn(batch_size, chunk, act_dim, generator=gen)
        time = torch.full((batch_size,), 0.5)

        if dtype_override is not None:
            images = [img.to(dtype_override) for img in images]
            actions = actions.to(dtype_override)
            noise = noise.to(dtype_override)
            time = time.to(dtype_override)

        return {
            "image_0": images[0],
            "image_1": images[1],
            "image_2": images[2],
            "img_mask_0": img_masks[0],
            "img_mask_1": img_masks[1],
            "img_mask_2": img_masks[2],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "actions": actions,
            "noise": noise,
            "time": time,
        }
