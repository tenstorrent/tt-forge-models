# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from collections import deque
from torch import Tensor
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.modeling_pi05 import (
    make_att_2d_masks,
    clone_past_key_values,
)
from types import MethodType


@torch.no_grad()
def _sample_actions_consistent(policy, images, img_masks, tokens, masks, noise):
    """Dtype-consistent reimplementation of ``PI05Pytorch.sample_actions``.

    Lerobot's stock ``sample_actions`` / ``denoise_step`` were only exercised
    at float32: they hardcode an fp32 timestep, build the additive attention
    mask as fp32 (``torch.where(..., 0.0, BIG)``), and force ``suffix_out`` to
    fp32 before the action projection. When the model is run in bfloat16 (the
    only precision that fits the n150's 12 GB device DRAM) those fp32 tensors
    collide with bf16 weights and the graph fails to compile.

    This routine mirrors the exact math of the stock loop but keeps every
    floating-point tensor at the model's parameter dtype, so the model runs
    cleanly in a single precision. Run at fp32 it is numerically identical to
    the stock path; run at bf16 it is what the device executes.
    """
    model = policy.model
    cfg = model.config
    # Derive the working dtype from an input rather than model.parameters():
    # load_inputs builds the floating-point inputs at the model dtype, and
    # reading it from a tensor keeps torch.compile/dynamo from tracing into
    # ``.parameters()`` (which raises an InternalTorchDynamoError).
    dtype = noise.dtype
    device = tokens.device
    bsize = tokens.shape[0]

    num_steps = cfg.num_inference_steps

    # --- prefix (vision + language) forward, cached ---
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks
    )
    prefix_embs = prefix_embs.to(dtype)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.long), dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks).to(
        dtype
    )
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
        "eager"
    )

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    # --- flow-matching denoising loop ---
    dt = -1.0 / num_steps
    x_t = noise.to(dtype)
    for step in range(num_steps):
        time = 1.0 + step * dt
        timestep = torch.tensor(time, dtype=dtype, device=device).expand(bsize)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
            x_t, timestep
        )
        suffix_embs = suffix_embs.to(dtype)

        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            bsize, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.long), dim=-1)[:, None]
        position_ids = (
            prefix_offsets + torch.cumsum(suffix_pad_masks.to(torch.long), dim=1) - 1
        )

        full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks).to(
            dtype
        )
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"
        )

        pkv = clone_past_key_values(past_key_values)
        outputs_embeds, _ = model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=pkv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1][:, -cfg.chunk_size :].to(dtype)
        v_t = model.action_out_proj(suffix_out)
        x_t = x_t + dt * v_t

    return x_t


@torch.no_grad()
def forward(
    self,
    images: Tensor,
    img_masks: Tensor,
    tokens: Tensor,
    masks: Tensor,
    noise: Tensor = None,
    **kwargs,
) -> Tensor:
    """
    Forward pass replicating ``select_action`` queue behavior for Pi-0.5.

    On the first call (or when the queue is drained), runs the model's
    ``sample_actions`` flow-matching denoising loop to produce a chunk of
    actions and fills a queue. Subsequent calls pop the next action from
    that queue without re-running the model.

    Unlike Pi-0, Pi-0.5 encodes the robot state inside the language tokens,
    so ``sample_actions`` takes only (images, img_masks, tokens, masks);
    there is no separate ``state`` argument.

    Queues are kept **per-device** so that the test harness (which calls
    forward once on CPU and once on the TT device using the same model
    instance) gets independent queues -- the CPU run never leaks actions
    into the TT run or vice-versa.

    Args:
        images (Tensor): Preprocessed image tensors (one per camera view).
        img_masks (Tensor): Masks indicating valid image views.
        tokens (Tensor): Tokenized language observations (state-encoded).
        masks (Tensor): Attention masks for the language tokens.
        noise (Tensor, optional): Pre-generated noise tensor for deterministic
            flow-matching sampling. When provided, both CPU and device runs use
            the same starting noise, ensuring a reproducible PCC comparison.
        **kwargs: Additional keyword arguments passed to ``sample_actions``.

    Returns:
        Tensor: A single action from the model (shape: [batch_size, action_dim]).
    """
    if not hasattr(self, "_device_queues"):
        self._device_queues = {}

    device_key = str(tokens.device)
    if device_key not in self._device_queues:
        self._device_queues[device_key] = deque()

    queue = self._device_queues[device_key]
    if len(queue) == 0:
        actions = _sample_actions_consistent(
            self, images, img_masks, tokens, masks, noise
        )

        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = actions[:, : self.config.n_action_steps]

        queue.extend(actions.transpose(0, 1))

    return queue.popleft()


def get_custom_pi05_policy(
    pretrained_model_name: str, config_dtype: str = None
) -> PI05Policy:
    """
    Create a customized Pi-0.5 Policy instance for inference.

    This function:
    1. Loads the original PI05Policy from a pretrained model name.
    2. Overrides the ``forward`` method with a custom inference-forward
       function that drives the flow-matching ``sample_actions`` loop.

    Args:
        pretrained_model_name (str): The name or path of the pretrained Pi-0.5 model.
        config_dtype (str, optional): When set (e.g. "bfloat16"), the model is
            *built* at this precision so the ~14.5 GB fp32 instance is never
            materialized. This is essential on memory-constrained hosts: the
            4B-param model otherwise peaks well above 30 GB during loading +
            compilation. The fp32 safetensors are still cast into the bf16
            parameters on load.

    Returns:
        PI05Policy: An instance of the Pi-0.5 Policy with an overridden
                    inference ``forward`` method.
    """
    if config_dtype is not None:
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(pretrained_model_name)
        config.dtype = config_dtype
        policy = PI05Policy.from_pretrained(pretrained_model_name, config=config)
    else:
        policy = PI05Policy.from_pretrained(pretrained_model_name)
    policy.forward = MethodType(forward, policy)
    return policy
