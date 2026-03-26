# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from collections import deque
from torch import Tensor
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
from types import MethodType


@torch.no_grad()
def preprocess_for_sampling(self, batch: dict[str, Tensor]):
    """
    Preprocess a batch for action sampling during inference.

    This method extracts and prepares all inputs required by the
    inference-time `forward` method, replicating the preprocessing
    logic originally used inside `select_action`.

    Args:
        batch (dict[str, Tensor]): Dictionary containing environment observations.

    Returns:
        images (Tensor): Preprocessed image tensors.
        img_masks (Tensor): Masks indicating valid pixels in images.
        lang_tokens (Tensor): Tokenized language observations.
        lang_masks (Tensor): Attention masks for language tokens.
        state (Tensor): State vector including proprioception / joint positions.
    """
    if (
        "observation.images.base_0_rgb" in self.config.image_features
        and "observation.images.left_wrist_0_rgb" in self.config.image_features
    ):
        batch["observation.images.base_0_rgb"] = batch.pop("observation.images.image")
        batch["observation.images.left_wrist_0_rgb"] = batch.pop(
            "observation.images.image2"
        )

    images, img_masks = self._preprocess_images(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    state = self.prepare_state(batch)

    return images, img_masks, lang_tokens, lang_masks, state


_original_cumsum = torch.cumsum


def _safe_cumsum(input, dim, **kwargs):
    """torch.cumsum wrapper that casts bool -> long before calling cumsum.

    StableHLO reduce_window on bool inputs fails during TTIR legalization.
    Casting bool to int64 first avoids this while being numerically lossless.
    """
    if input.dtype == torch.bool:
        input = input.to(torch.long)
    return _original_cumsum(input, dim, **kwargs)


@torch.no_grad()
def _sample_actions_with_graph_break(
    model,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise=None,
    num_steps=None,
    **kwargs,
):
    """Replicated sample_actions with an XLA graph break after embed_prefix.

    The StableHLO -> TTIR compiler hangs when the full embed_prefix graph
    (SigLIP vision encoder + language embedding + concat) is composed with
    make_att_2d_masks (which broadcasts [1, N] -> [1, N, N]) in a single
    compilation unit.  Inserting xm.mark_step() between them forces two
    separate, smaller compilations that each succeed.
    """
    if num_steps is None:
        num_steps = model.config.num_inference_steps

    bsize = state.shape[0]
    device = state.device

    if noise is None:
        actions_shape = (
            bsize,
            model.config.chunk_size,
            model.config.max_action_dim,
        )
        noise = model.sample_noise(actions_shape, device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )

    # --- XLA graph break ---
    # Without this, the combined graph (embed_prefix + make_att_2d_masks)
    # hangs during StableHLO -> TTIR compilation.  Each subgraph compiles
    # fine individually.  No-op on CPU (safe for reference run).
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except ImportError:
        pass

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=time_tensor,
        )
        x_t = x_t + dt * v_t

    return x_t


@torch.no_grad()
def forward(
    self,
    images: Tensor,
    img_masks: Tensor,
    lang_tokens: Tensor,
    lang_masks: Tensor,
    state: Tensor,
    noise: Tensor = None,
    **kwargs
) -> Tensor:
    """
    Forward pass replicating ``select_action`` queue behavior.

    On the first call (or when the queue is drained), runs
    ``sample_actions`` to produce a chunk of actions and fills a queue.
    Subsequent calls pop the next action from that queue without
    re-running the model.

    Queues are kept **per-device** so that the test harness (which calls
    forward once on CPU and once on the TT device using the same model
    instance) gets independent queues -- the CPU run never leaks actions
    into the TT run or vice-versa.

    Args:
        images (Tensor): Preprocessed image tensors.
        img_masks (Tensor): Masks for images.
        lang_tokens (Tensor): Tokenized language observations.
        lang_masks (Tensor): Attention masks for language tokens.
        state (Tensor): State vector including proprioception / joint positions.
        noise (Tensor, optional): Pre-generated noise tensor for deterministic
            diffusion sampling. When provided, both CPU and device runs use
            the same starting noise, ensuring reproducible PCC comparison.
        **kwargs: Additional keyword arguments passed to `sample_actions`.

    Returns:
        Tensor: A single action from the model (shape: [batch_size, action_dim]).
    """
    if not hasattr(self, "_device_queues"):
        self._device_queues = {}

    device_key = str(state.device)
    if device_key not in self._device_queues:
        self._device_queues[device_key] = deque()

    queue = self._device_queues[device_key]
    if len(queue) == 0:
        torch.cumsum = _safe_cumsum
        try:
            actions = _sample_actions_with_graph_break(
                self.model, images, img_masks, lang_tokens, lang_masks,
                state, noise=noise, **kwargs
            )
        finally:
            torch.cumsum = _original_cumsum

        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = actions[:, : self.config.n_action_steps]

        queue.extend(actions.transpose(0, 1))

    return queue.popleft()


def get_custom_pi0_policy(pretrained_model_name: str) -> PI0Policy:
    """
    Create a customized Pi-0 Policy instance for inference.

    This function:
    1. Loads the original PI0Policy from a pretrained model name.
    2. Adds `preprocess_for_sampling` method to preprocess
       inputs for action sampling.
    3. Overrides the `forward` method with a custom inference-forward
       that includes the XLA graph break workaround.

    Args:
        pretrained_model_name (str): The name or path of the pretrained Pi-0 model.

    Returns:
        PI0Policy: An instance of the Pi-0 Policy with overridden
                   inference methods.
    """
    policy = PI0Policy.from_pretrained(pretrained_model_name)
    policy.preprocess_for_sampling = MethodType(preprocess_for_sampling, policy)
    policy.forward = MethodType(forward, policy)
    return policy
